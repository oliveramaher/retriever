import re
import numpy as np
from rank_bm25 import BM25Okapi

def bm25_scores(passages, query):
    def keyword_tokens(text):
        return re.findall(r"\w+", text.lower()) # Separates into a list of words

    words = [keyword_tokens(passage["text"]) for passage in passages]
    return BM25Okapi(words).get_scores(keyword_tokens(query))

def combine_rankings(semantic_scores, keyword_scores):
    combined = np.zeros(len(semantic_scores))

    for rank, i in enumerate(np.argsort(-semantic_scores), start=1):
        combined[i] += 1 / (60 + rank)
    for rank, i in enumerate(np.argsort(-keyword_scores), start=1):
        if keyword_scores[i] > 0: combined[i] += 1 / (60 + rank)

    return np.argsort(-combined)[:10]
