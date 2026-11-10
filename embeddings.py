import numpy as np

MODEL = "jinaai/jina-embeddings-v5-text-nano"

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
