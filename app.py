import json
from pathlib import Path
import numpy as np

from embeddings import MODEL, load_model, encode
from segmentation import tokenizer, extract_text, chunk_text
from rankings import bm25_scores, combine_rankings

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

def search(query):
    folder = Path(".booksearch")
    if not (folder / "model.txt").exists() or (folder / "model.txt").read_text(encoding="utf-8") != MODEL:
        raise ValueError("Uses a different embedding model")

    vectors = np.load(folder / "vectors.npy")

    with open(folder / "passages.json") as file:
        passages = json.load(file)

    query_vector = encode(load_model(), [query], query=True)[0]
    semantic_scores = vectors @ query_vector
    keyword_scores = bm25_scores(passages, query)
    best_matches = combine_rankings(semantic_scores, keyword_scores)

    for i in best_matches:
        print(passages[i]["text"], end="\n\n")
