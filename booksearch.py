import argparse
import json
from pathlib import Path
import numpy as np

def tokenizer():
    import tiktoken
    return tiktoken.get_encoding('cl100k_base')

def extract_pages(pdf_path):
    import pymupdf
    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            yield page.number + 1, page.get_text('text') # Return each page # + text

def chunk_pages(pages, tokenizer, size=250, overlap=50):
    passages = []
    for page_number, text in pages:
        tokens = tokenizer.encode(text) # Tokenize
        
        for start in range(0, len(tokens), size - overlap):
            text_chunk = tokenizer.decode(tokens[start : start + size]) # Get tokenized text
            if text_chunk.strip(): # If text exists
                passages.append({ # Add page # + text
                    "page": page_number,
                    "text": text_chunk,
                })
            if start + size >= len(tokens): # Breaks once its larger
                break
                
    return passages

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


def index_book(pdf_path):
    passages = chunk_pages(extract_pages(pdf_path), tokenizer())
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
        passage = passages[i]
        print(f"Page {passage['page']}")
        print(passage["text"], end="\n\n")

def main():
    # Define CLI, takes in command: index or search, then a pdf or query
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('index').add_argument('pdf') # Index method takes a PDF
    commands.add_parser('search').add_argument('query') # Query method takes a str
    args = parser.parse_args()
    if args.command == 'index':
        index_book(args.pdf)
    else:
        search(args.query)

main()