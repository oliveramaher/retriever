import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

MODEL = "jinaai/jina-reranker-v3.5"
CACHE = ".booksearch/huggingface/hub"

def load_reranker():
    # The model is downloaded once; searching uses the local copy.
    folder = snapshot_download(MODEL, cache_dir=CACHE, local_files_only=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained(folder, trust_remote_code=True,
                                      local_files_only=True, dtype=torch.float32,
                                      attn_implementation="sdpa")
    return model.to(device).eval()

def rerank(query, passages, model, limit=10):
    if not passages:
        return []
    results = model.rerank(query, [p["text"] for p in passages], top_n=limit)
    # Use the returned indexes to keep each original passage and its metadata.
    return [passages[int(result["index"])] for result in results]
