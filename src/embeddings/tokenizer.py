"""The embedding model's own tokenizer, used to size chunks so the embedder never truncates them.

Loaded from the model files fastembed already downloaded (`tokenizer.json`), without opening an
ONNX session. Truncation and padding are switched off so `count` reports the true length
(special tokens included, as the embedder adds them).
"""
import hashlib

from fastembed import TextEmbedding
from fastembed.common.preprocessor_utils import load_tokenizer


class EmbedTokenizer:
    def __init__(self, model_id: str):
        self.model_id = model_id
        model = TextEmbedding(model_name=model_id, lazy_load=True)
        model_dir = model.model._model_dir
        self._sha1 = hashlib.sha1((model_dir / "tokenizer.json").read_bytes()).hexdigest()
        tokenizer, _ = load_tokenizer(model_dir)
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._tok = tokenizer

    def info(self) -> dict:
        """Recorded in the index manifest."""
        return {"model": self.model_id, "tokenizer_json_sha1": self._sha1,
                "special_overhead": self.special_overhead}

    def count(self, text: str) -> int:
        return len(self._tok.encode(text).ids)

    def count_many(self, texts: list[str]) -> list[int]:
        return [len(e.ids) for e in self._tok.encode_batch(texts)]

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of every real token (special tokens excluded)."""
        enc = self._tok.encode(text)
        return [
            (s, e)
            for (s, e), special in zip(enc.offsets, enc.special_tokens_mask, strict=True)
            if not special
        ]

    @property
    def special_overhead(self) -> int:
        """Tokens the model adds around every text (e.g. [CLS] and [SEP])."""
        return len(self._tok.encode("").ids)
