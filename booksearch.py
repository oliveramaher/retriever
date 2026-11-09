import argparse
import json
import re
from pathlib import Path
import numpy as np
import spacy
from rank_bm25 import BM25Okapi

MODEL = "jinaai/jina-embeddings-v5-text-nano"

def tokenizer():
    import tiktoken
    return tiktoken.get_encoding('cl100k_base')

def extract_text(txt_path):
    return Path(txt_path).read_text(encoding="utf-8") # Return entire text as str

def chunk_text(text, tokenizer, size=250):
    nlp = spacy.load("en_core_web_sm")
    sentences = list(nlp(text).sents)

    chunks, previous = [], None
    if not sentences:
        return []
    start_chunk = sentences[0].start_char

    for curr in sentences:
        next_chunk = text[start_chunk : curr.end_char]

        if previous and len(tokenizer.encode(next_chunk)) > size:
            end_chunk = previous.end_char
            chunks.append({
                "text": text[start_chunk:end_chunk],
                "text_position": (start_chunk, end_chunk),
            })

            start_chunk = previous.start_char  # Start at the beginning of last sentence to overlap

        previous = curr

    end_chunk = previous.end_char
    chunks.append({
        "text": text[start_chunk:end_chunk],
        "text_position": (start_chunk, end_chunk),
    })

    return chunks

def load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(MODEL, trust_remote_code=True, device=device,
                               cache_folder=".booksearch/huggingface/hub",
                               model_kwargs={"dtype": torch.float32})

def encode(model, texts, query=False):
    prompt = "query" if query else "document"
    return np.asarray(model.encode(texts, task="retrieval", prompt_name=prompt,
                                  batch_size=8, normalize_embeddings=True,
                                  show_progress_bar=not query), dtype=np.float32)


def index_book(txt_path):
    passages = chunk_text(extract_text(txt_path), tokenizer())
    texts = [passage["text"] for passage in passages]
    vectors = encode(load_model(), texts)

    folder = Path(".booksearch")
    folder.mkdir(exist_ok=True)
    np.save(folder / "vectors.npy", vectors)
    with open(folder / "passages.json", "w") as file:
        json.dump(passages, file)
    (folder / "model.txt").write_text(MODEL, encoding="utf-8")

def keyword_tokens(text):
    return re.findall(r"\w+", text.lower()) # Separates into a list of words

def bm25_scores(passages, query):
    words = [keyword_tokens(passage["text"]) for passage in passages]
    return BM25Okapi(words).get_scores(keyword_tokens(query))

def combine_rankings(semantic_scores, keyword_scores):
    combined = np.zeros(len(semantic_scores))

    for rank, i in enumerate(np.argsort(-semantic_scores), start=1):
        combined[i] += 1 / (60 + rank)
    for rank, i in enumerate(np.argsort(-keyword_scores), start=1):
        if keyword_scores[i] > 0: combined[i] += 1 / (60 + rank)

    return np.argsort(-combined)[:10]

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

def main():
    # Define CLI, takes in command: index or search, then a pdf or query
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('index').add_argument('txt') # Index method takes a .txt
    commands.add_parser('search').add_argument('query') # Query method takes a str
    args = parser.parse_args()
    if args.command == 'index':
        index_book(args.txt)
    else:
        search(args.query)

if __name__ == "__main__":
    main()
