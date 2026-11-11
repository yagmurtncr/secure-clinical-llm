"""Turkish PII detectors. Every detector returns spans; validators cut false positives (checksums, mod-97)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Protocol


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    text: str


def tckn_valid(s: str) -> bool:
    """Turkish national ID (T.C. Kimlik No) checksum."""
    if len(s) != 11 or not s.isdigit() or s[0] == "0":
        return False
    d = [int(c) for c in s]
    d10 = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    return d[9] == d10 and d[10] == sum(d[:10]) % 10


def iban_valid(s: str) -> bool:
    s = s.replace(" ", "").upper()
    if not re.fullmatch(r"TR\d{24}", s):
        return False
    num = "".join(str(int(c, 36)) for c in s[4:] + s[:4])
    return int(num) % 97 == 1


_TCKN = re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")
_IBAN = re.compile(r"\bTR\d{2}(?: ?\d{4}){5} ?\d{2}\b", re.I)
_PHONE = re.compile(r"(?<![\d+])(?:\+90[ -]?|0090[ -]?|0)?\(?(5\d{2}|[2-4]\d{2})\)?[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}(?!\d)")
_PHONE_CTX = re.compile(r"(?i)(tel|telefon|gsm|cep|ileti[şs]im|phone)\W*$")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# key: value fields typical for Turkish lab / discharge reports
_NAME_FIELD = re.compile(
    r"(?im)^[ \t]*(?:hasta(?:n[ıi]n)?[ \t]+ad[ıi](?:[ \t]+soyad[ıi])?|ad[ıi]?[ \t]*soyad[ıi]?|ad[ \t]*-[ \t]*soyad|hasta|"
    r"doktor|dr\.?|hekim|isteyen[ \t]+hekim|onaylayan)[ \t]*[:\-][ \t]*(?:dr\.?[ \t]*)?"
    r"(?P<v>[A-ZÇĞİÖŞÜ][a-zçğıöşüA-ZÇĞİÖŞÜ'.]+(?:[ \t]+[A-ZÇĞİÖŞÜ][a-zçğıöşüA-ZÇĞİÖŞÜ'.]+){0,3})")
_ID_FIELD = re.compile(r"(?im)\b(?:protokol|dosya|hasta|barkod|örnek|ornek)\s*(?:no|numaras[ıi])\s*[:\-]\s*(?P<v>[A-Z0-9-]{4,})")
_DOB_FIELD = re.compile(r"(?im)\bdo[ğg]um\s+tarihi\s*[:\-]\s*(?P<v>\d{1,2}[./-]\d{1,2}[./-]\d{2,4})")
_ADDRESS_FIELD = re.compile(r"(?im)^[ \t]*adres\s*[:\-]\s*(?P<v>.+)$")


def _regex_spans(text: str) -> List[Span]:
    spans: List[Span] = []
    for m in _IBAN.finditer(text):
        if iban_valid(m.group()):
            spans.append(Span(m.start(), m.end(), "IBAN", m.group()))
    for m in _TCKN.finditer(text):
        if tckn_valid(m.group()):
            spans.append(Span(m.start(), m.end(), "TCKN", m.group()))
    for m in _EMAIL.finditer(text):
        spans.append(Span(m.start(), m.end(), "EMAIL", m.group()))
    for m in _PHONE.finditer(text):
        g = m.group()
        formatted = g.startswith(("0", "+", "(")) or any(c in g for c in " -")
        # a bare 10-digit run (5xxxxxxxxx) is often a serial / lot number: require a phone keyword nearby
        if formatted or _PHONE_CTX.search(text[max(0, m.start() - 25):m.start()]):
            spans.append(Span(m.start(), m.end(), "TELEFON", g))
    for rx, label in ((_NAME_FIELD, "AD"), (_ID_FIELD, "KAYIT_NO"), (_DOB_FIELD, "DOGUM_TARIHI"),
                      (_ADDRESS_FIELD, "ADRES")):
        for m in rx.finditer(text):
            v = m.group("v").rstrip(" .")
            s = m.start("v")
            spans.append(Span(s, s + len(v), label, v))
    return spans


class NerBackend(Protocol):
    def __call__(self, text: str) -> List[Span]: ...


class HFNer:
    """Optional transformer NER (person / location / organization) for names in free text."""

    LABELS = {"PER": "AD", "PERSON": "AD", "LOC": "ADRES", "ORG": "KURUM"}

    def __init__(self, model: str = "akdeniz27/bert-base-turkish-cased-ner", min_score: float = 0.80,
                 keep: Iterable[str] = ("AD",)):
        from transformers import pipeline

        self.pipe = pipeline("token-classification", model=model, aggregation_strategy="simple")
        self.min_score, self.keep = min_score, set(keep)

    def __call__(self, text: str) -> List[Span]:
        merged = []  # word-piece aggregation can split "Ayşe" / "Yılmaz": merge same-label neighbours
        for e in self.pipe(text):
            label = self.LABELS.get(e["entity_group"].upper())
            if label not in self.keep:
                continue
            if merged and merged[-1][2] == label and not text[merged[-1][1]:e["start"]].strip():
                s0, _, _, sc = merged[-1]
                merged[-1] = (s0, e["end"], label, max(sc, float(e["score"])))
            else:
                merged.append((e["start"], e["end"], label, float(e["score"])))
        return [Span(a, b, lab, text[a:b]) for a, b, lab, sc in merged if sc >= self.min_score]


_PRIORITY = {"IBAN": 0, "TCKN": 1, "EMAIL": 2, "KAYIT_NO": 3, "DOGUM_TARIHI": 4, "TELEFON": 5, "AD": 6,
             "ADRES": 7, "KURUM": 8}


def detect(text: str, ner: Optional[Callable[[str], List[Span]]] = None) -> List[Span]:
    """All PII spans, overlaps resolved (longer span first, then label priority)."""
    spans = _regex_spans(text) + (ner(text) if ner else [])
    spans.sort(key=lambda s: (-(s.end - s.start), _PRIORITY.get(s.label, 99), s.start))
    chosen: List[Span] = []
    for s in spans:
        if all(s.end <= c.start or s.start >= c.end for c in chosen):
            chosen.append(s)
    return sorted(chosen, key=lambda s: s.start)
