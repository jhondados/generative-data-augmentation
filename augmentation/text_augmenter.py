"""LLM text data augmenter."""
from langchain_google_vertexai import ChatVertexAI
from typing import List
import random

AUGMENT_STRATEGIES = [
    "paraphrase using different vocabulary while preserving meaning",
    "simplify the language while keeping the same intent",
    "make it more formal and professional",
    "rephrase using synonyms and different sentence structure",
    "rewrite from a different perspective but same content",
]

class TextAugmenter:
    def __init__(self, n_augmentations: int = 5):
        self.llm = ChatVertexAI(model_name="gemini-1.5-flash-002", temperature=0.7)
        self.n = n_augmentations

    def augment(self, text: str, label: str = "", n: int = None) -> List[str]:
        n = n or self.n
        strategies = random.sample(AUGMENT_STRATEGIES, min(n, len(AUGMENT_STRATEGIES)))
        augmented = []
        for strategy in strategies:
            prompt = f"""Rewrite the following text: {strategy}.
Original: {text}
Rewritten version (single sentence/paragraph, same length):"""
            aug = self.llm.invoke(prompt).content.strip()
            if aug and aug != text and len(aug) > 10: augmented.append(aug)
        return augmented

    def augment_dataset(self, texts: List[str], labels: List[str],
                        target_per_class: int = 1000) -> tuple:
        from collections import Counter
        class_counts = Counter(labels)
        aug_texts, aug_labels = list(texts), list(labels)
        for cls, count in class_counts.items():
            if count < target_per_class:
                cls_texts = [t for t, l in zip(texts, labels) if l == cls]
                needed = target_per_class - count
                while needed > 0:
                    for text in cls_texts:
                        if needed <= 0: break
                        new_texts = self.augment(text, cls, n=min(3, needed))
                        aug_texts.extend(new_texts)
                        aug_labels.extend([cls] * len(new_texts))
                        needed -= len(new_texts)
        return aug_texts, aug_labels
