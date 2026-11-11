"""Reversible pseudonymization: PII -> stable placeholders like [AD_1]; the mapping never leaves the gateway."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .detectors import Span, detect


@dataclass
class MaskResult:
    text: str
    mapping: Dict[str, str] = field(repr=False)  # placeholder -> original value
    spans: List[Span] = field(repr=False)

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for k in self.mapping:
            label = k[1:].rsplit("_", 1)[0]
            out[label] = out.get(label, 0) + 1
        return out


def _propagate(text: str, spans: List[Span]) -> List[Span]:
    """A value found once (e.g. the name in the header field) is masked everywhere it appears verbatim."""
    taken = [(s.start, s.end) for s in spans]
    extra = []
    for s in {(s.label, s.text) for s in spans if len(s.text) >= 3}:
        for m in re.finditer(r"(?<!\w)" + re.escape(s[1]) + r"(?!\w)", text):
            if all(m.end() <= a or m.start() >= b for a, b in taken):
                extra.append(Span(m.start(), m.end(), s[0], s[1]))
                taken.append((m.start(), m.end()))
    return sorted(spans + extra, key=lambda x: x.start)


def mask(text: str, ner: Optional[Callable] = None) -> MaskResult:
    spans = _propagate(text, detect(text, ner))
    mapping: Dict[str, str] = {}
    reverse: Dict[Tuple[str, str], str] = {}
    per_label: Dict[str, int] = {}
    pieces, last = [], 0
    for s in spans:
        key = (s.label, s.text)
        if key not in reverse:
            per_label[s.label] = per_label.get(s.label, 0) + 1
            ph = f"[{s.label}_{per_label[s.label]}]"
            reverse[key], mapping[ph] = ph, s.text
        pieces += [text[last:s.start], reverse[key]]
        last = s.end
    pieces.append(text[last:])
    return MaskResult("".join(pieces), mapping, spans)


_PH = re.compile(r"\[([A-Z_]+_\d+)\]")


def unmask(text: str, mapping: Dict[str, str]) -> str:
    """Restore placeholders the model repeated; unknown placeholders are left untouched."""
    return _PH.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
