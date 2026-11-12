"""Estimate claim support and trim unnecessary sentences from passage edges."""
import os

# Keep the model’s custom Python code in the project cache.
os.environ.setdefault("HF_MODULES_CACHE", ".booksearch/huggingface/modules")

import torch
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder
from segmentation import nlp

MODEL = 'dleemiller/finecat-nli-ultra-l'
CACHE = '.booksearch/huggingface/hub'


class FactChecker:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        folder = snapshot_download(MODEL, cache_dir=CACHE, local_files_only=True,
                                   allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py"])
        self.model = CrossEncoder(folder, trust_remote_code=True, device=self.device)
        labels = self.model.model.config.id2label
        self.entailment = next(int(i) for i, label in labels.items() if label.lower() == 'entailment')

    def score(self, evidence, claim):
        # Premise first, claim second. The model returns entailment, neutral, contradiction.
        tokens = self.model.tokenizer(evidence, claim, truncation=False)['input_ids']
        if len(tokens) > self.model.max_seq_length:
            raise ValueError('Passage and claim exceed the NLI model input limit.')
        probabilities = self.model.predict([(evidence, claim)], apply_softmax=True)
        return float(probabilities[0][self.entailment])


def trim_passage(passage, claim, score, threshold=0.5):
    """Trim one sentence at a time; scores estimate support, not proof."""
    text = passage['text']
    sentences = [s for s in nlp(text).sents if s.text.strip()]
    full_score = score(text, claim)
    result = {'text': text, 'text_position': passage.get('text_position'),
              'score': full_score, 'supported': full_score >= threshold}
    if not sentences or not result['supported']:
        return result

    start, end = 0, len(sentences)
    while end - start > 1:
        choices = []
        for new_start, new_end in ((start + 1, end), (start, end - 1)):
            left = sentences[new_start].start_char
            right = sentences[new_end - 1].end_char
            value = score(text[left:right], claim)
            if value >= threshold:
                choices.append((value, new_start, new_end))
        if not choices:
            break
        value, start, end = max(choices)
        left, right = sentences[start].start_char, sentences[end - 1].end_char
        result.update(text=text[left:right], score=value)
        if passage.get('text_position') is not None:
            offset = passage['text_position'][0]
            result['text_position'] = (offset + left, offset + right)
    return result


def find_supported(passages, claim, score, threshold=0.5, limit=3):
    """Return the first passage that meets the support threshold."""
    for rank, passage in enumerate(passages[:limit], start=1):
        result = trim_passage(passage, claim, score, threshold)
        if result['supported']:
            result['rank'] = rank
            return result
    return None
