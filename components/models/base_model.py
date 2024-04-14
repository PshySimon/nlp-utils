import torch
from torch import nn
import torch.nn.functional as F

class BeamHypothesis:
    def __init__(self, n_hyp, max_length, length_penalty, early_stopping):
        self.max_length = max_length
        self.n_hyp = n_hyp
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.hyp)
    
    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(score, idx) for idx, (score, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
    
    def is_done(self, best_sum_logprobs):
        # 当前容器中beam数量比num_beams要少时，说明还没完成
        if len(self) < self.n_hyp:
            return False
        elif self.early_stopping:
            return True
        else:
            # 
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty
        

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



# model 
class BaseModel(nn.Module):
    def __init__(self, model_config) -> None:
        super().__init__()
        self.max_length = model_config.max_length
        self.tokenizer = model_config.tokenizer
        self.temperature = model_config.temperature
        self.top_k = model_config.top_k
        self.top_p = model_config.top_p
        self.repetition_penalty = model_config.repetition_penalty
        self.pad_token_id = model_config.pad_token_id
        self.eos_token_ids = model_config.eos_token_ids
        self.length_penalty = model_config.length_penalty
        self.num_beams = model_config.num_beams
        self.vocab_size = model_config.vocab_size
        self.do_sample = model_config.do_sample
    
    def _prepare_generation_inputs(self, input_ids):
        return {
            "input_ids": input_ids
        }


    def generate(self,
                 input_ids,
                 cur_len,
                 batch_size
                 ):
        if self.num_beams == 1:
            return self.greedy_search(
                      input_ids,
                      cur_len,
                      self.max_length,
                      self.do_sample,
                      self.temperature,
                      self.top_p,
                      self.top_k,
                      self.repetition_penalty,
                      self.pad_token_id,
                      self.eos_token_ids,
                      batch_size
            )
        else:
            return self.beam_search(
                    input_ids,
                    cur_len,
                    self.max_length,
                    self.do_sample,
                    self.temperature,
                    self.top_k,
                    self.top_p,
                    self.repetition_penalty,
                    self.pad_token_id,
                    self.eos_token_ids,
                    batch_size,
                    self.length_penalty,
                    self.num_beams,
                    self.vocab_size
            )
            
    # 批量贪心解码
    @torch.no_grad()
    def greedy_search(self, 
                      input_ids,
                      cur_len,
                      max_length,
                      do_sample,
                      temperature,
                      top_p,
                      top_k,
                      repetition_penalty,
                      pad_token_id,
                      eos_token_ids,
                      batch_size):
        unfinished_sents = input_ids.new(batch_size).fill_(1)

        while cur_len < max_length:
            # input_ids = [batch_size, seq_length]
            model_inputs = self._prepare_generation_inputs(input_ids)
            # logits = [batch_size, seq_length, vocab_size]
            logits, _ = self(**model_inputs)

            # next_token_logits = [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]
            
            # TODO  do_output_past
            # TODO  repetition penalty
            if do_sample:
                # TODO do sample
                pass
            else:
                # next_token = [batch_size]
                next_token = torch.argmax(next_token_logits, dim=-1)

            # tokens_to_add = [batch_size]
            tokens_to_add = next_token * unfinished_sents + pad_token_id * (1 - unfinished_sents)
            # input_ids = [batch_size, seq_length + 1]
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            # judge whether the some of sent should stop generation
            for eos_token_id in eos_token_ids:
                # tensor.ne returns bool type, should transform into long type
                unfinished_sents.mul_(tokens_to_add.ne(eos_token_id).long())
            cur_len += 1

            if unfinished_sents.max() == 1:
                break
        
        if cur_len == max_length:
            input_ids.masked_fill_(unfinished_sents.to(dtype=torch.bool), eos_token_ids[0])
        return input_ids
    
    @torch.no_grad()
    def beam_search(self,
                    input_ids,
                    cur_len,
                    max_length,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    pad_token_id,
                    eos_token_ids,
                    batch_size,
                    length_penalty,
                    num_beams,
                    vocab_size):
        # input_ids = [batch_size, cur_len]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, cur_len)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, cur_len)

        # 用于记录每个样本的beam容器
        generated_hyps = [
            BeamHypothesis(num_beams, max_length, length_penalty, early_stopping=False)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        # 每个句子都会有num_beams个分数，表示cur_len长度下各个路径的打分
        # 预测下一个单词时，通过计算当前beam中所表示的路径与整个词表的组合概率
        # [batch_size, num_beams * vocab_size]
        # 每个sentence从候选的num_beams * vocab_size中抽取num_beams条路径
        # 作为下一个单词的beam_scores
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        # 最开始输入的单词都是bos_token，那么打分都会是一样的
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            # input_ids = [batch_size * num_beams, cur_len]
            model_inputs = self._prepare_generation_inputs(input_ids)
            # logits = [batch_size * num_beams, cur_len]
            logits, _ = self(**model_inputs)
            # scores = [batch_size * num_beams, vocab_size]
            scores = logits[:, -1, :]

            # TODO do ouput past
            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for previous_token in set(input_ids[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if scores[i, previous_token] < 0:
                            scores[i, previous_token] *= repetition_penalty
                        else:
                            scores[i, previous_token] /= repetition_penalty
            # TODO do sample
            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                scores = top_k_top_p_filtering(
                    scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # Sample 2 next words for each beam (so we have some spare tokens and match output of greedy beam search)
                next_words = torch.multinomial(F.softmax(scores, dim=-1), num_samples=2)  # (batch_size * num_beams, 2)
                # Compute next scores
                _scores = F.log_softmax(scores, dim=-1)  # (batch_size * num_beams, vocab_size)
                _scores = torch.gather(_scores, -1, next_words)  # (batch_size * num_beams, 2)
                next_scores = _scores + beam_scores[:, None].expand_as(_scores)  # (batch_size * num_beams, 2)
                # Match shape of greedy beam search
                next_words = next_words.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
                next_scores = next_scores.view(batch_size, 2 * num_beams)  # (batch_size, 2 * num_beams)
            else:
                # scores = [batch_size * num_beams, vocab_size]
                score = F.log_softmax(scores, dim=-1)
                assert scores.size() == (batch_size * num_beams, vocab_size)
                # beam_scores[:, None] = [batch_size * num_beams, 1]
                # 这里其实就是把上一步的路径打分和当前的路径打分
                # vocab里的每个word对应的log_prob都加上上一步的打分
                # 现在计算出当前time_step对应所有word的路径打分
                _scores = scores + beam_scores[:, None].expand_as(scores)
                _scores = _scores.view(batch_size, num_beams * vocab_size)
                # 选取topk个，这里选取2 * num_beams个候选beam
                # 这里为了便于并行化计算，矩阵被重排列成num_beams * vocab_size了
                # 要想取word_id和beam_id其实也不难，实际上相当于把二维数组reshape成了一维
                # 所以根据数组的相对位置，每个sentence记录的是连续的vocab_size个打分
                # 每vocab_size个打分为一组，这一组就代表了前面的num_beams条路径到当前某个word的路径打分
                # 所以要想计算word_id，直接cur_pos % vocab_size，取余之后就能得到对应word的打分了
                # beam_id同理，cur_pos // vocab_size
                # next_scores存储的是beam_scores打分，next_words存储的是当前time_step的word_id
                next_scores, next_words = torch.topk(
                    _scores, 2 * num_beams, dim=1, largest=True, sorted=True)
                
            assert next_scores.size() == next_words.size() == (batch_size, 2 * num_beams)

            next_batch_beam = []

            for batch_ex in range(batch_size):
                # 获取下一个time_step的beam_scores中最大的分数与worst_scores作比较
                # worst_score是一个动态值，为当前打分中最低的路径得分
                # 如果当前预测的路径分数比历史最低值还要小，那么就表明已完成；否则还需要继续
                done[batch_ex] = done[batch_ex] or generated_hyps[batch_ex].is_done(next_scores[batch_ex].max().item())
                if done[batch_ex]:
                    # 已经完成的batch直接用pad_token填充
                    next_batch_beam.extend([0, pad_token_id, 0] * num_beams)
                    continue
                next_sent_beam = []

                for idx, score in zip(next_words[batch_ex], next_scores[batch_ex]):
                    beam_id = idx  // vocab_size
                    word_id = idx % vocab_size

                    # 如果预测出的结果是结束标志符或者已达到长度上限，直接添加到beam容器里
                    # idx, beam_id 和 batch_ex * num_beams + beam_id之间的关系：
                    # idx为按[batch_1_beam_1_word_1, batch_1_beam_1_word_2, ...,
                    # batch_1_beam_1_word_v, ... , batch_n_beam_m_word_v]排列的序号
                    # beam_id为上述序号中某个样本序列中的beam标号
                    # batch_ex * num_beams + beam_id则是第batch_ex个样本的beam_id个beam标号
                    # 把当前生成的input_ids保存起来，存到优先队列里，根据打分来去掉最差的路径
                    if word_id.item() in eos_token_ids or cur_len + 1 == max_length:
                        generated_hyps[batch_ex].add(
                            input_ids[batch_ex * num_beams + beam_id, :cur_len].clone(), score.item())
                    else:
                        # 保存对应beam_id、打分和word_id
                        next_sent_beam.append((score, word_id, batch_ex * num_beams + beam_id))

                    if len(next_sent_beam) == num_beams:
                        break

                assert len(next_sent_beam) == 0 if cur_len + 1 == max_length else num_beams
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, pad_token_id, 0)] * num_beams 
                # 下一个step的全部样本beam信息全部保存下来，不包括已经结束生成的情况
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_ex + 1)

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            # re-order batch
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_words.unsqueeze(1)], dim=-1)

            # update current length
            cur_len = cur_len + 1

            # stop when we are done with each sentence
            if all(done):
                break

        # select the best hypotheses
        tgt_len = input_ids.new(batch_size)
        best = []

        for i, hypotheses in enumerate(generated_hyps):
            best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
            tgt_len[i] = len(best_hyp) + 1  # +1 for the <EOS> symbol
            best.append(best_hyp)

        # generate target batch
        decoded = input_ids.new(batch_size, tgt_len.max().item()).fill_(pad_token_id)
        for i, hypo in enumerate(best):
            decoded[i, : tgt_len[i] - 1] = hypo
            decoded[i, tgt_len[i] - 1] = eos_token_ids[0]

        return decoded
