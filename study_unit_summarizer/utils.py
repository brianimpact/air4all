import re

import numpy as np
import torch
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge


def cosine_similarity(v1, v2, t):
    return ((v1 * v2).sum() / (v1.norm() * v2.norm() * 2 + 1e-8) + 0.5).cpu().item()

def exponential_similarity(v1, v2, t):
    return torch.exp(-(v1 - v2).norm() / t * np.log(2)).cpu().item()

def inverse_similarity(v1, v2, t):
    return torch.pow(1 + (v1 - v2).norm() / t, -1).cpu().item()

def simple_preprocess(text):
    text = text.strip()
    text = text[0].upper() + text[1:].lower()
    text = text.replace('(', ' (').replace(')', ') ')
    text = re.sub('\s+', ' ', text).replace(' .', '.')
    if not text.endswith('.'):
        text = text + '.'
    return text

def calc_rouge(s1, s2):
    rouge = Rouge()
    score = (rouge.get_scores(s1, s2)[0]['rouge-l']['f'] + rouge.get_scores(s2, s1)[0]['rouge-l']['f']) / 2
    return score

def calc_bleu(s1, s2):
    bleu_weights = (1/4, 1/4, 1/4, 1/4)
    score = (corpus_bleu([s1.split()], [s2.split()], weights=bleu_weights) +  corpus_bleu([s2.split()], [s1.split()], weights=bleu_weights)) / 2
    return score

def calc_meteor(s1, s2):
    score = (meteor_score([s1], s2) + meteor_score([s2], s1)) / 2
    return score

def intercluster_similarity(cl1, cl2, fn):
    values = []
    for el1 in cl1:
        for el2 in cl2:
            values.append(fn(el1, el2))
    return sum(values) / len(values)

def cluster_centroid(cl, fn):
    if len(cl) == 1:
        return cl[0]
    else:
        values = [[] for _ in cl]
        for i in range(len(cl) - 1):
            for j in range(i + 1, len(cl)):
                score = fn(cl[i], cl[j])
                values[i].append(score)
                values[j].append(score)
        values = [sum(v) / len(v) for v in values]
        return cl[sorted(list(range(len(cl))), key=lambda x: values[x])[-1]]

def make_string(ranked_list):
    add_flag = True
    up_to_500 = ''
    entire_str = ''
    for l in ranked_list:
        if len(up_to_500 + ' ' + l[1].strip()) > 500:
            add_flag = False
        elif add_flag:
            up_to_500 = up_to_500 + ' ' + l[1].strip()
        entire_str += '%.6f\t%s\n' % (l[0], l[1])
    return up_to_500, entire_str
