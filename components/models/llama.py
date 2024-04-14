import math
import torch
import torch.nn as nn
from .base_model import BaseModel


class LlamaConfig:
    def __init__(self) -> None:
        # -----------model_parameters---------------------
        self.padding_idx = 0
        self.hidden_size = 2048
        self.vocab_size = 32000
        self.num_hidden_layers = 22
        self.rms_norm_eps = 1e-05
        self.device = torch.device("cpu")
        # -----------embedding_parameters-----------------
        self.max_position_embeddings = 2048
        self.rope_theta = 10000
        self.attention_bias = False
        # -----------attention_parameters-----------------
        self.num_heads = 32
        self.attention_dropout = 0.0
        self.num_key_value_heads = 4
        # -----------mlp_parameters-----------------------
        self.intermidiate_size = 5632
        # -----------generation_parameters----------------
        self.eos_token_id = self.padding_idx
        self.pad_token_id = self.padding_idx
        self.max_generate_len = 1024
        self.generation_algorithm = "greedy_search"


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def _init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)   \
                          .float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len = max_position_embeddings,
            device = self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # [seq_length]一维向量，存储的是[0,1,2,3,4,...,seq_length]
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # inv_freq就是系数，维度是[dim]的一维向量，存储的是[1/(base^(2i)/d)]
        # 两者做外积也就是元素两两相乘，形成[seq_length, dim]的矩阵，那么就可以得出，每个seq_length，dim都有对应的position embedding
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(self, x, seq_len=None):
        # 针对超出训练时的句子长度，临时扩充生成对应的embedding
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype)
        )
    

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch_size, num_key_value_heads * n_rep, seq_len, head_dim)


class LlamaAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads //  self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert (self.head_dim * self.num_heads) == self.hidden_size

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta
        )

    def forward(self,
                hidden_states,
                attention_mask,
                position_ids,
                last_step_kv_states=None,
                use_cache=False):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if last_step_kv_states is not None:
            kv_seq_len += last_step_kv_states[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if last_step_kv_states  is not None:
            key_states = torch.cat([last_step_kv_states[0], key_states], dim=2)
            value_states = torch.cat([last_step_kv_states[1], value_states], dim=2)
        
        last_step_kv_states = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attention_mask + attn_weights
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, last_step_kv_states
    
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermidiate_size = config.intermidiate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermidiate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermidiate_size, bias=False)
        self.down_proj = nn.Linear(self.intermidiate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMlp(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self,
                hidden_states,
                attention_mask,
                position_ids,
                last_step_kv_states=None,
                use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weight, last_step_kv_states = self.self_attn(
            hidden_states, attention_mask, position_ids, last_step_kv_states, use_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weight, last_step_kv_states

class LlamaModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.padding_idx = config.padding_idx
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.init_weights()

    def _initialize_weights(self, module):
        _init_weights(module)

    def init_weights(self):
        self.apply(self._initialize_weights)

    def _expand_mask(self, mask, device, dtype, tgt_len=None):
        # 生成padding_mask，用户传入的是[batch_size, seq_len_with_past]
        # 需要变成[batch_size, 1, tgt_len, tgt_len + past_kv_length]
        batch_size, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len
        expandded_mask = mask[:, None, None, :]                    \
                         .expand(batch_size, 1, tgt_len, src_len)  \
                         .to(dtype=dtype)
        inverted_mask = 1.0 - expandded_mask
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min).to(device)

    def _make_causal_mask(self, input_shape, dtype, device, past_kv_length=0):
        bsz, tgt_len = input_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        # 下三角全为0，上三角包括对角线都是-inf
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        if past_kv_length > 0:
            mask = torch.cat([
                torch.zeros(tgt_len, past_kv_length, dtype=dtype, device=device),
                mask
            ], dim=-1)
        mask = mask.to(dtype)
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_kv_length)

    def create_attn_mask(self,
                         attention_mask,
                         input_shape,
                         input_embeds,
                         past_kv_length):
        # attn_mask分为两部分，一部分是causal_mask，另一部分是padding_mask
        # attention_mask分为两种情况，一种是训练时的mask，一种是生成时的mask
        # 注意padding_mask只有在批量生成时会用到，单个样本用不到这个
        # 训练时，不需要逐个token生成，因此past_kv_length是0，因此需要padding_mask和causal_mask
        # 生成时，需要逐个生成token，在此情况下分为两种情况
        #       情况一：提示词的attention_mask，和训练时一样，需要padding_mask和causal_mask
        #       情况二：生成token与前面的提示词拼接，这时causal mask不需要了，因为上一个生成的token
        #             后面的token是一定要被前面看到，所以当传入前面生成的token时，mask必定是1
        combined_attention_mask = None

        dtype = input_embeds.dtype
        device = input_embeds.device

        # 长度大于1才会有causal_mask
        if input_shape[-1] > 1:
            combined_attention_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device,
                past_kv_length
            )
        
        if attention_mask is not None:
            expanded_mask = self._expand_mask(
                attention_mask,
                device,
                dtype,
                tgt_len=input_shape[-1]
            )
            combined_attention_mask = (combined_attention_mask + expanded_mask) \
                if expanded_mask is not None else combined_attention_mask 
        
        return combined_attention_mask

    def forward(self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                last_step_kv_states_list=None,
                use_cache=False):
        batch_size, seq_length = input_ids.shape
        next_kv_states_list = []
        all_attentions = []

        seq_length_with_past = seq_length
        past_kv_length = 0
        if last_step_kv_states_list is not None:
            past_kv_length = last_step_kv_states_list[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_kv_length

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(
                past_kv_length,
                seq_length + past_kv_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = input_embeds

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=input_embeds.device
            )

        attention_mask = self.create_attn_mask(
            attention_mask,
            (batch_size, seq_length),
            input_embeds,
            past_kv_length
        )

        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_output = decoder_layer(
                hidden_states,
                attention_mask,
                position_ids,
                last_step_kv_states_list[layer_idx] if last_step_kv_states_list is not None else None,
                use_cache=use_cache
            )
            hidden_states = layer_output[0]

            if use_cache:
                next_kv_states_list.append(layer_output[2])
            
            all_attentions.append(layer_output[1])
        
        hidden_states = self.norm(hidden_states)
        return hidden_states, all_attentions, next_kv_states_list


class LlamaForCausalLM(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.init_weights()
    
    def _initialize_weights(self, module):
        _init_weights(module)

    def init_weights(self):
        self.apply(self._initialize_weights)

    def forward(self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                labels=None,
                last_step_kv_states_list=None,
                use_cache=False):
        hidden_states, all_attentions, present_kv_states_list = self.model(
            input_ids,
            attention_mask,
            position_ids,
            last_step_kv_states_list=last_step_kv_states_list,
            use_cache=use_cache
        )
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            labels = labels.type(torch.LongTensor)
            # Shift so that tokens < n predict n 
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return logits, loss

