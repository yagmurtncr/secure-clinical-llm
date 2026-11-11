"""The request pipeline: mask -> flag labs -> prompt -> split inference -> unmask -> grounding check."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch

from ..clinical.grounding import check
from ..clinical.labs import parse_labs
from ..clinical.prompt import build_messages
from ..pii.masker import mask, unmask
from ..split.runtime import SplitPipeline


@dataclass
class Tokenizer:
    """Minimal interface so the engine works with any HF tokenizer (and with fakes in tests)."""

    encode_chat: Callable[[list], List[int]]
    decode: Callable[[Sequence[int]], str]
    eos_token_ids: Sequence[int]

    @classmethod
    def from_hf(cls, tok) -> "Tokenizer":
        def enc(messages):
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys"):  # BatchEncoding in newer transformers
                ids = ids["input_ids"]
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            return [int(i) for i in ids]

        eos = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"),
                           tok.convert_tokens_to_ids("<|eot_id|>")} if isinstance(i, int) and i >= 0]
        return cls(enc, lambda ids: tok.decode(ids, skip_special_tokens=True), eos)


class Engine:
    def __init__(self, pipeline: SplitPipeline, tokenizer: Tokenizer, ner: Optional[Callable] = None,
                 max_new_tokens: int = 512, max_prompt_tokens: int = 3072):
        self.pipeline, self.tok, self.ner = pipeline, tokenizer, ner
        self.max_new_tokens, self.max_prompt_tokens = max_new_tokens, max_prompt_tokens

    def analyze(self, report: str) -> dict:
        t0 = time.perf_counter()
        masked = mask(report, self.ner)
        labs = parse_labs(masked.text)
        t_mask = time.perf_counter() - t0
        ids = self.tok.encode_chat(build_messages(masked.text, labs))
        if len(ids) > self.max_prompt_tokens:
            raise ValueError(f"report too long ({len(ids)} tokens > {self.max_prompt_tokens})")
        new_ids, stats = self.pipeline.generate(torch.tensor([ids]), self.max_new_tokens, self.tok.eos_token_ids)
        raw_answer = self.tok.decode(new_ids)
        grounding = check(raw_answer, masked.text, labs)
        return {
            "answer": unmask(raw_answer, masked.mapping),
            "labs": [l.to_dict() for l in labs],
            "grounding": grounding.to_dict(),
            "pii": masked.counts(),
            "masked_report": masked.text,
            "timing": {"mask_s": round(t_mask, 4), "prefill_s": round(stats.prefill_s, 3),
                       "decode_s": round(stats.decode_s, 3), "tokens_per_s": round(stats.tokens_per_s, 2),
                       "prompt_tokens": stats.prompt_tokens, "new_tokens": stats.new_tokens,
                       "stage_s": [round(s, 3) for s in stats.stage_s]},
        }
