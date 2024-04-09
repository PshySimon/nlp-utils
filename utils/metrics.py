import numpy as np
from typing import Union
from collections import Counter


def get_bleu4_score(reference: Union[str, list[str]], outputs: Union[str, list[str]], n_gram: int=4) -> float:
    '''
    获取bleu4分数
    '''
    
    weights = np.ones(n_gram) * (1.0 / n_gram)

    outputs_len, reference_len = len(outputs), len(reference)

    if not type(reference) is list:
        reference = list(reference)
    if not type(outputs) is list:
        outputs = list(outputs)

    outputs_counter = get_n_grams(outputs, n_gram=n_gram)
    reference_counter = get_n_grams(reference, n_gram=n_gram)

    ngram_counter_clip = outputs_counter & reference_counter

    clip_counter = np.zeros(n_gram)
    output_ngram_counter = np.zeros(n_gram)

    for (_, ngram), cnt in ngram_counter_clip.items():
        clip_counter[ngram - 1] += cnt 
    
    for (_, ngram), cnt in outputs_counter.items():
        output_ngram_counter[ngram - 1] += cnt
    
    # 一旦有一个值为零，计算对数时结果就是-inf，那么求和也是-inf，再取指数就是0
    if np.min(clip_counter) == 0.0:
        return np.array(0.0)

    precision_scores = clip_counter / output_ngram_counter
   
    # bleu
    log_precision_scores = weights * np.log(precision_scores)
    
    # 几何平均形式求平均值然后加权
    geometric_mean = np.exp(np.sum(log_precision_scores))
    brevity_penalty = np.exp(1 - (reference_len / outputs_len))
    bleu = brevity_penalty * geometric_mean

    return bleu


def get_n_grams(sentence, n_gram):
    n = len(sentence)
    ngram_counter = Counter()
    for j in range(1, n_gram + 1):
        for i in range(n - j + 1):
            key = " ".join(sentence[i:i + j])
            ngram_counter[(key, j)] += 1
    return ngram_counter
