"""Structured answers: the model explains, the code decides.

The free-text mode asks the model to say whether each value is high or low and checks afterwards. Here the
model returns JSON with one plain-language explanation per test, and the gateway renders the answer itself:
the value, range and status in every line come from ``parse_labs`` and are never generated. Explanation
sentences that the grounding checker flags (a contradicting direction word or an invented number) are removed
before the answer is shown. If the JSON cannot be parsed, the answer falls back to the computed table alone.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .grounding import _DIRECTION, check
from .labs import LabResult
from .prompt import flags_table

SYSTEM = (
    "Sen bir laboratuvar sonucu açıklama asistanısın. Hastaya sade Türkçe açıklama yazarsın.\n"
    "Yalnızca geçerli JSON döndür, başka metin yazma. Şema:\n"
    '{"ozet": "2-3 cümlelik genel özet", "testler": [{"test": "<tablodaki test adı>", '
    '"aciklama": "bu testin ne ölçtüğü ve bu sonucun ne anlama gelebileceği, 1-2 cümle"}], '
    '"oneri": "hekime danışma önerisi, 1 cümle"}\n'
    "Kurallar: durumları (yüksek/düşük/normal) tablodan al, değiştirme. Yeni sayı, tanı veya ilaç yazma. "
    "Köşeli parantezli yer tutucuları (ör. [AD_1]) olduğu gibi bırak."
)

STATUS_TR = {"yüksek": "Yüksek", "düşük": "Düşük", "normal": "Normal"}


def build_structured_messages(masked_report: str, labs: List[LabResult]) -> list:
    user = (f"Tahlil raporu:\n```\n{masked_report.strip()}\n```\n\nHesaplanan durumlar:\n{flags_table(labs)}\n\n"
            "Her test için JSON şemasına uygun açıklama yaz.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def _loads_lenient(s: str):
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    start = s.find("{")
    if start < 0:
        return None
    depth, end, in_str, esc = 0, None, False, False
    for i, ch in enumerate(s[start:], start):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None
    body = re.sub(r",\s*([}\]])", r"\1", s[start:end])  # trailing commas
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


_ITEM = re.compile(r'\{\s*"test"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"aciklama"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')
_OZET = re.compile(r'"ozet"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _salvage(raw: str) -> Optional[dict]:
    """Output cut off by the token limit: keep the test objects that were completed."""
    items = [{"test": json.loads(f'"{a}"'), "aciklama": json.loads(f'"{b}"')} for a, b in _ITEM.findall(raw)]
    if not items:
        return None
    oz = _OZET.search(raw)
    return {"ozet": json.loads(f'"{oz.group(1)}"') if oz else "", "testler": items, "oneri": ""}


def parse_structured(raw: str) -> Optional[dict]:
    obj = _loads_lenient(raw)
    if not isinstance(obj, dict):
        obj = _salvage(raw)
    if not isinstance(obj, dict):
        return None
    tests = obj.get("testler")
    if not isinstance(tests, list):
        return None
    clean = []
    for t in tests:
        if isinstance(t, dict) and isinstance(t.get("test"), str) and isinstance(t.get("aciklama"), str):
            clean.append({"test": t["test"].strip(), "aciklama": t["aciklama"].strip()})
    return {"ozet": str(obj.get("ozet", "")).strip(), "testler": clean, "oneri": str(obj.get("oneri", "")).strip()}


def _match(name: str, labs: List[LabResult]) -> Optional[LabResult]:
    key = name.lower().split(" (")[0].strip()
    for l in labs:
        base = l.name.lower().split(" (")[0]
        if key == base or key.startswith(base) or base.startswith(key):
            return l
    return None


def _directions(sent: str) -> set:
    low = sent.lower()
    return {v for k, v in _DIRECTION.items() if re.search(rf"\b{k}", low)}


def _sanitize(text: str, source: str, labs: List[LabResult], removed: list,
              subject: Optional[LabResult] = None) -> str:
    """Drop sentences the checker flags; keep the rest verbatim. Inside one test's explanation a sentence that
    names no test ("Bu değer yüksek…") is about that test (``subject``), so its direction word is checked too."""
    keep = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sent:
            continue
        g = check(sent, source, labs)
        bad = [c["test"] for c in g.contradictions]
        if subject is not None and not g.checked_statements:
            dirs = _directions(sent)
            if len(dirs) == 1 and dirs != {subject.status}:
                bad.append(subject.name)
        if not g.unsupported_numbers and not bad:
            keep.append(sent)
        else:
            removed.append({"sentence": sent, "unsupported_numbers": g.unsupported_numbers, "contradictions": bad})
    return " ".join(keep)


def _drop_restatement(text: str, lab: LabResult) -> str:
    """Small models often echo the flags line ("180.5 ng/mL (referans 15-150) → yüksek") instead of explaining.
    The rendered line already shows that, so echoed sentences are dropped as redundant (not as errors)."""
    keep = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
            if s and "→" not in s and not s.strip().startswith(lab.raw_value)]
    return " ".join(keep)


@dataclass
class StructuredAnswer:
    text: str
    parsed: bool
    explained: int
    total: int
    removed_sentences: List[dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"parsed": self.parsed, "explained_tests": self.explained, "total_tests": self.total,
                "removed_sentences": len(self.removed_sentences)}


def render(raw: str, masked_source: str, labs: List[LabResult]) -> StructuredAnswer:
    parsed = parse_structured(raw)
    removed: list = []
    expl: Dict[str, str] = {}
    if parsed:
        for t in parsed["testler"]:
            l = _match(t["test"], labs)
            if l is not None and l.name not in expl:
                expl[l.name] = _sanitize(_drop_restatement(t["aciklama"], l), masked_source, labs, removed, subject=l)
    abnormal = [l for l in labs if l.status != "normal"]
    normal = [l for l in labs if l.status == "normal"]
    lines = []
    if parsed and parsed["ozet"]:
        s = _sanitize(parsed["ozet"], masked_source, labs, removed)
        if s:
            lines += [s, ""]
    for title, group in (("Referans dışı sonuçlar", abnormal), ("Referans içindeki sonuçlar", normal)):
        if not group:
            continue
        lines.append(f"**{title}**")
        for l in group:
            unit = f" {l.unit}" if l.unit else ""
            line = f"- **{l.name}: {l.raw_value}{unit}** (referans {l.low:g}–{l.high:g}) → {STATUS_TR[l.status]}"
            if expl.get(l.name):
                line += f". {expl[l.name]}"
            lines.append(line)
        lines.append("")
    oneri = _sanitize(parsed["oneri"], masked_source, labs, removed) if parsed and parsed["oneri"] else ""
    lines.append(oneri or "Bu sonuçlar tanı yerine geçmez; değerlendirme için hekiminize danışın.")
    return StructuredAnswer("\n".join(lines).strip(), parsed is not None, sum(1 for v in expl.values() if v),
                            len(labs), removed)
