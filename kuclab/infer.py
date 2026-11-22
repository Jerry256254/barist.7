from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .evaluate import load_checkpoint, predict_full, predict_proba_chunked

CAT_PRIORITY = ["sexual_abuse", "self_harm", "violence", "weapons_explosives", "drugs",
                "cybercrime_fraud", "jailbreak_injection", "illegal_marketplace", "other_unsafe",
                "hate_harassment"]  # hate last: catch-all


def _cat_rank(c: str) -> int:
    return CAT_PRIORITY.index(c) if c in CAT_PRIORITY else 99

from .langdetect import detect_lang, must_catch_hit, review_hit, is_support_context, identity_positive, SUPPORT_LINK


@dataclass
class Verdict:
    label: str
    probability: float
    threshold: float
    categories: list[str] | None = None
    chunks: int = 1
    needs_review: bool = False

    @property
    def is_unsafe(self) -> bool:
        return self.label in ("1", "unsafe")

    def as_dict(self) -> dict:
        d = {"label": self.label, "unsafe": self.is_unsafe, "score": round(self.probability, 4)}
        if self.categories:
            d["categories"] = self.categories
        if getattr(self, "needs_review", False):
            d["needs_review"] = True
        if getattr(self, "override", None):
            d["override"] = self.override
        if getattr(self, "needs_support", None):
            d["needs_support"] = self.needs_support
        if self.chunks > 1:
            d["chunks"] = self.chunks
        return d


class SafetyModel:
    """Inference wrapper: text in, '0' or '1' out."""

    def __init__(self, checkpoint: str | Path = "checkpoints/best.pt", threshold: float = 0.5, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.tokenizer, self.model_config, self.state = load_checkpoint(checkpoint, self.device)
        self.max_chars = int(self.state.get("max_chars", 500))
        self.max_len = self.model_config.max_seq_len
        self.threshold = threshold
        self.temperature = float(self.state.get("temperature", 1.0) or 1.0)
        self.must_catch_override = dict(self.state.get("must_catch_override") or {})
        # Barist.5: category-head hard block off until the head passes its own recall gate
        self.category_block = bool(getattr(self.model_config, "category_block", False) or self.state.get("category_block", False))

    def _cats(self) -> list[str]:
        from .categories import CATEGORIES

        return CATEGORIES if len(self.model_config.answer_ids) > 2 else []

    def predict(self, text: str) -> Verdict:
        probs = predict_proba_chunked(self.model, self.tokenizer, [text[: self.max_chars * 8]], self.max_len, self.device, temperature=self.temperature)[0]
        cats = self._cats()
        row = predict_full(self.model, self.tokenizer, [text[: self.max_chars]], self.max_len, self.device, temperature=self.temperature)[0]
        lang = detect_lang(text)
        lang_thr = self.state.get("lang_thresholds") or {}
        thr = float(lang_thr.get(lang, lang_thr.get("default", self.threshold)))
        is_unsafe = probs >= thr
        hit = must_catch_hit(text)  # Barist.1 hard pre-filter (CSAM / self-harm / DIY / poison / market)
        review = review_hit(text)   # jailbreak patterns -> needs_review, not a hard block
        if hit:
            is_unsafe = True
        help_ctx = is_support_context(text)
        support = help_ctx or "self_harm" in hit
        cat_scores = dict(zip(cats, row[2:])) if cats else {}
        # Barist.7: model self_harm >= 0.3 -> needs_support (never a block by itself)
        if cats and "self_harm" in cats and cat_scores["self_harm"] >= 0.3:
            support = True
        # Hard block: sexual_abuse / self_harm >= 0.3 blocks regardless of language threshold
        # (0.3 is a BLOCK decision only, not a label threshold).
        # Barist.5: gated behind category_block flag (head recall gate, default OFF).
        cat_block: list[str] = []
        if self.category_block and cats and not help_ctx:  # help-other frames are never hard-blocked
            cat_block = [c for c in ("sexual_abuse", "self_harm") if cat_scores.get(c, 0.0) >= 0.3]
            if cat_block:
                is_unsafe = True
        if help_ctx and not hit and not cat_block:
            # help-seeking context (e.g. "jak pomoct kamarádovi…") is NEVER blocked
            is_unsafe = False
        if identity_positive(text) and not hit:
            is_unsafe = False  # Barist.7: identity-positive waiver
        flagged = None
        if cats and is_unsafe:
            _fired = [c for c, p in zip(cats, row[2:]) if p >= self.threshold]
            flagged = sorted(set(_fired) | set(hit) | set(cat_block), key=lambda c: (_cat_rank(c), -cat_scores.get(c, 1.0)))
            if not flagged:
                # unsafe must always carry >=1 category: top-1 by score
                flagged = [max(cats, key=lambda c: cat_scores[c])]
        v = Verdict(label="unsafe" if is_unsafe else "safe", probability=probs,
                    threshold=thr, categories=flagged)
        v.override = "must_catch" if hit else ("category_block" if cat_block else None)
        v.needs_support = SUPPORT_LINK if support else None
        v.needs_review = bool(review)
        return v

    def predict_batch(self, texts: list[str]) -> list[Verdict]:
        trimmed = [t[: self.max_chars * 8] for t in texts]
        probs = predict_proba_chunked(self.model, self.tokenizer, trimmed, self.max_len, self.device, temperature=self.temperature)
        return [
            Verdict(label="unsafe" if p >= self.threshold else "safe", probability=p, threshold=self.threshold)
            for p in probs
        ]
