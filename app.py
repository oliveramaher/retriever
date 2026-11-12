import json
from pathlib import Path
import numpy as np

from embeddings import MODEL, load_model, encode
from segmentation import tokenizer, extract_text, chunk_text
from rankings import bm25_scores, combine_rankings
from reranker import load_reranker, rerank

def index_book(txt_path):
    model = load_model()
    passages = chunk_text(extract_text(txt_path), tokenizer())
    texts = [passage["text"] for passage in passages]
    vectors = encode(model, texts)

    folder = Path(".booksearch")
    folder.mkdir(exist_ok=True)
    np.save(folder / "vectors.npy", vectors)
    with open(folder / "passages.json", "w") as file:
        json.dump(passages, file)
    (folder / "model.txt").write_text(MODEL, encoding="utf-8")

def search(query, verify=False):
    folder = Path(".booksearch")
    if not (folder / "model.txt").exists() or (folder / "model.txt").read_text(encoding="utf-8") != MODEL:
        raise ValueError("Uses a different embedding model")

    vectors = np.load(folder / "vectors.npy")

    with open(folder / "passages.json") as file:
        passages = json.load(file)

    query_vector = encode(load_model(), [query], query=True)[0]
    semantic_scores = vectors @ query_vector
    keyword_scores = bm25_scores(passages, query)
    best_matches = combine_rankings(semantic_scores, keyword_scores, limit=20)
    candidates = [passages[i] for i in best_matches]
    results = rerank(query, candidates, load_reranker())

    if verify:
        from nli import FactChecker, find_supported
        checker = FactChecker()
        result = find_supported(results, query, checker.score)
        if result is None:
            print("Not found in the first 3 passages (support not established).")
        else:
            print(f"Estimated support: {result['score']:.3f} | Result {result['rank']}")
            print(result['text'])
    else:
        for passage in results:
            print(passage["text"], end="\n\n")
