"""Post-hoc checks on the model answer: unsupported numbers and contradicted high/low statements."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .labs import LabResult

_NUM = re.compile(r"(?<![\w\]])\d+(?:[.,]\d+)?(?![\w\[])")
_DIRECTION = {
    "yüksek": "yüksek", "yükselmiş": "yüksek", "artmış": "yüksek", "fazla": "yüksek",
    "düşük": "düşük", "azalmış": "düşük", "eksik": "düşük",
    "normal": "normal", "sınırlar içinde": "normal", "referans aralığında": "normal",
}
_PH = re.compile(r"\[[A-Z_]+_\d+\]")


def _norm(x: str) -> str:
    v = float(x.replace(",", "."))
    return f"{v:g}"


@dataclass
class GroundingReport:
    unsupported_numbers: List[str] = field(default_factory=list)
    contradictions: List[dict] = field(default_factory=list)
    checked_statements: int = 0

    @property
    def ok(self) -> bool:
        return not self.unsupported_numbers and not self.contradictions

    def to_dict(self) -> dict:
        return {"ok": self.ok, "unsupported_numbers": self.unsupported_numbers,
                "contradictions": self.contradictions, "checked_statements": self.checked_statements}


def check(answer: str, source: str, labs: List[LabResult]) -> GroundingReport:
    rep = GroundingReport()
    text = _PH.sub(" ", answer)
    allowed = {_norm(n) for n in _NUM.findall(_PH.sub(" ", source))}
    for l in labs:
        allowed |= {f"{l.value:g}", f"{l.low:g}", f"{l.high:g}"}
    for n in _NUM.findall(text):
        if _norm(n) not in allowed and _norm(n) not in rep.unsupported_numbers:
            rep.unsupported_numbers.append(_norm(n))
    # sentence-level: a sentence that names exactly one test and one direction word is a checkable claim
    for sent in re.split(r"(?<=[.!?])\s+|\n+", answer):
        low = sent.lower()
        named = [l for l in labs if l.name.lower().split(" (")[0] in low]
        dirs = {v for k, v in _DIRECTION.items() if re.search(rf"\b{k}", low)}
        if len(named) == 1 and len(dirs) == 1:
            rep.checked_statements += 1
            said, truth = dirs.pop(), named[0].status
            if said != truth:
                rep.contradictions.append({"test": named[0].name, "said": said, "actual": truth, "sentence": sent.strip()})
    return rep
