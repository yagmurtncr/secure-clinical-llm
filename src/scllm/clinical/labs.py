"""Parse lab result rows and flag them against their reference ranges, deterministically (no LLM involved)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import List, Optional

_NUM = r"[<>]?\d+(?:[.,]\d+)?"
_ROW = re.compile(
    rf"^\s*(?P<name>[A-Za-zÇĞİÖŞÜçğıöşü][\wÇĞİÖŞÜçğıöşü .()/%+-]*?)\s*[:\t ]\s*(?P<value>{_NUM})\s*"
    rf"(?P<unit>[A-Za-zµμ%/^0-9.³]+(?:/[A-Za-zµμ0-9.³^]+)?)?\s*"
    rf"[\[(]?\s*(?P<low>{_NUM})\s*[-–]\s*(?P<high>{_NUM})\s*[\])]?\s*(?P<flag>[HLhl*↑↓])?\s*$")


def _f(s: str) -> float:
    return float(s.lstrip("<>").replace(",", "."))


@dataclass
class LabResult:
    name: str
    value: float
    unit: Optional[str]
    low: float
    high: float
    raw_value: str

    @property
    def status(self) -> str:
        if self.value < self.low:
            return "düşük"
        if self.value > self.high:
            return "yüksek"
        return "normal"

    @property
    def deviation(self) -> float:
        """Signed distance outside the range, as a fraction of the range width (0 inside the range)."""
        width = (self.high - self.low) or 1.0
        if self.value < self.low:
            return (self.value - self.low) / width
        if self.value > self.high:
            return (self.value - self.high) / width
        return 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "status": self.status, "deviation": round(self.deviation, 3)}


def parse_labs(text: str) -> List[LabResult]:
    out = []
    for line in text.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        name = re.sub(r"\s+", " ", m.group("name")).strip(" :")
        low, high = _f(m.group("low")), _f(m.group("high"))
        if low > high or len(name) < 2:
            continue
        out.append(LabResult(name, _f(m.group("value")), m.group("unit"), low, high, m.group("value")))
    return out
