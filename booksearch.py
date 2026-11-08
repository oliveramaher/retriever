import argparse
import json
from pathlib import Path
import numpy as np
import spacy

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
    
    model_folder = Path("qwen_model")

    if not model_folder.exists():
        model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        model.save(str(model_folder))
    return SentenceTransformer(str(model_folder), local_files_only=True, model_kwargs={"dtype": torch.float32})

def encode(model, texts, query=False):
    if query:
        prompt="Instruction: Given a question about a book, retrieve passages that answer the question.\nQuestion: "
        texts = [prompt + text for text in texts]
    return np.asarray(model.encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=not query), dtype=np.float32)


def index_book(txt_path):
    passages = chunk_text(extract_text(txt_path), tokenizer())
    texts = [passage["text"] for passage in passages]
    vectors = encode(load_model(), texts)

    folder = Path(".booksearch")
    folder.mkdir(exist_ok=True)
    np.save(folder / "vectors.npy", vectors)
    with open(folder / "passages.json", "w") as file:
        json.dump(passages, file)

def search(query):
    folder = Path(".booksearch")

    vectors = np.load(folder / "vectors.npy")

    with open(folder / "passages.json") as file:
        passages = json.load(file)

    query_vector = encode(load_model(), [query], query=True)[0]
    scores = vectors @ query_vector

    best_matches = np.argsort(scores)[::-1][:10]

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

main()
