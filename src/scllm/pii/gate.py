"""Fail-closed privacy gate: the last check before a masked report is sent to untrusted nodes.

The leakage study (eval/hidden_state_leakage.py) shows a node can read most input tokens back from the hidden
states it receives, so split inference by itself does not hide the report. The guarantee therefore rests on
masking. The gate re-scans the *masked* text with recall-first settings and refuses to send it if anything
still looks like PII. A refused report is never sent to the nodes; the caller gets an error and can review it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .detectors import Span, _regex_spans
from .masker import MaskResult

_PH = re.compile(r"\[[A-Z_]+_\d+\]")
# titles and honorifics are not names; the model sometimes tags them at low confidence next to a placeholder
_TITLES = {"dr", "uzm", "prof", "doç", "doc", "op", "hemş", "hems", "hemşire", "ebe", "sayın", "sn", "bey", "hanım",
           "hasta", "hekim"}
_LONG_DIGITS = re.compile(r"(?<!\d)\d(?:[ .-]?\d){9,}(?!\d)")  # 10+ digits, possibly grouped


@dataclass
class GateReport:
    ok: bool
    findings: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        # labels and positions only: the values themselves must not end up in logs
        return {"ok": self.ok, "findings": [{k: v for k, v in f.items() if k != "text"} for f in self.findings]}


class PrivacyGate:
    def __init__(self, ner: Optional[Callable[[str], List[Span]]] = None, ner_threshold: float = 0.30,
                 block_long_digit_runs: bool = True):
        self.ner, self.ner_threshold = ner, ner_threshold
        self.block_long_digit_runs = block_long_digit_runs

    def _outside_placeholders(self, text: str, spans) -> list:
        ph = [(m.start(), m.end()) for m in _PH.finditer(text)]
        return [s for s in spans if all(s[1] <= a or s[0] >= b for a, b in ph)]

    def check(self, masked: MaskResult) -> GateReport:
        text = masked.text
        found = []
        for s in self._outside_placeholders(text, [(s.start, s.end, s.label, s.text) for s in _regex_spans(text)]):
            found.append({"source": "regex", "label": s[2], "start": s[0], "end": s[1], "text": s[3]})
        if self.block_long_digit_runs:
            for m in _LONG_DIGITS.finditer(text):
                digits = re.sub(r"\D", "", m.group())
                if len(digits) in (10, 11):  # phone (5xx…) or ID-number length; lab values are never this long
                    found.append({"source": "digits", "label": "NUMBER", "start": m.start(), "end": m.end(),
                                  "text": m.group()})
        if self.ner is not None:
            pipe = getattr(self.ner, "pipe", None)
            if pipe is not None:  # HFNer: look at low-confidence entities too
                ents = [(e["start"], e["end"], str(e["entity_group"]).upper(), float(e["score"])) for e in pipe(text)]
                ents = [(a, b, lab, sc) for a, b, lab, sc in ents if lab in ("PER", "PERSON") and sc >= self.ner_threshold]
            else:
                ents = [(s.start, s.end, s.label, 1.0) for s in self.ner(text) if s.label == "AD"]
            for a, b, lab, sc in self._outside_placeholders(text, ents):
                word = text[a:b].strip(" '.,")
                if len(word) >= 2 and word.replace("İ", "i").replace("I", "ı").lower() not in _TITLES:
                    found.append({"source": "ner", "label": "AD", "start": a, "end": b, "score": round(sc, 3),
                                  "text": text[a:b]})
        return GateReport(not found, found)
