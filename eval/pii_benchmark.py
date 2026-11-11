"""Synthetic, labeled Turkish lab-report PII benchmark.

Generates reports with gold spans (header fields, free-text mentions, hard negatives such as invalid 11-digit
barcodes and device serials) and scores the detector with exact-span precision / recall / F1 per label.

    python eval/pii_benchmark.py --n 500 --seed 0 [--ner] --out results/pii.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scllm.pii.detectors import detect  # noqa: E402
from scllm.pii.masker import _propagate  # noqa: E402

FIRST = ["Ayşe", "Mehmet", "Fatma", "Mustafa", "Emine", "Ahmet", "Hatice", "Ali", "Zeynep", "Hüseyin", "Elif",
         "Hasan", "Merve", "İbrahim", "Şule", "Can", "Özge", "Murat", "Gül", "Ömer", "Çağla", "Yusuf", "Büşra",
         "Emre", "Nur Yağmur", "Ece", "Burak", "Derya", "Kerem", "İrem"]
LAST = ["Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Yıldırım", "Öztürk", "Aydın", "Özdemir",
        "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Kara", "Koç", "Kurt", "Özkan", "Şimşek", "Güneş", "Erdoğan"]
LABS = [("Hemoglobin", "g/dL", 12.0, 16.0), ("Glukoz (açlık)", "mg/dL", 70, 100), ("TSH", "mIU/L", 0.4, 4.0),
        ("Ferritin", "ng/mL", 15, 150), ("ALT", "U/L", 0, 41), ("Kreatinin", "mg/dL", 0.6, 1.2),
        ("LDL Kolesterol", "mg/dL", 0, 130), ("B12 Vitamini", "pg/mL", 200, 900), ("WBC", "10^3/µL", 4.0, 10.0)]


def rand_tckn(r: random.Random) -> str:
    d = [r.randint(1, 9)] + [r.randint(0, 9) for _ in range(8)]
    d10 = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    d.append(d10)
    d.append(sum(d) % 10)
    return "".join(map(str, d))


def invalid_11(r: random.Random) -> str:
    while True:
        s = str(r.randint(10**10, 10**11 - 1))
        from scllm.pii.detectors import tckn_valid
        if not tckn_valid(s):
            return s


def rand_iban(r: random.Random) -> str:
    bban = "".join(str(r.randint(0, 9)) for _ in range(22))
    num = int("".join(str(int(c, 36)) for c in bban + "TR00"))
    check = 98 - num % 97
    raw = f"TR{check:02d}{bban}"
    return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4))


def rand_phone(r: random.Random) -> str:
    a, b, c, d = r.randint(500, 559), r.randint(100, 999), r.randint(10, 99), r.randint(10, 99)
    return r.choice([f"0{a} {b} {c} {d}", f"+90 {a} {b} {c} {d}", f"0{a}{b}{c}{d}", f"({a}) {b} {c} {d}",
                     f"0{a}-{b}-{c}-{d}"])


class Doc:
    def __init__(self):
        self.parts, self.spans = [], []

    def add(self, s: str, label: str | None = None):
        start = sum(len(p) for p in self.parts)
        self.parts.append(s)
        if label:
            self.spans.append((start, start + len(s), label))

    @property
    def text(self):
        return "".join(self.parts)


def make_report(r: random.Random) -> Doc:
    d = Doc()
    name = f"{r.choice(FIRST)} {r.choice(LAST)}"
    doctor = f"{r.choice(FIRST)} {r.choice(LAST)}"
    d.add(r.choice(["HASTA BİLGİLERİ\n", "LABORATUVAR SONUÇ RAPORU\n", ""]))
    d.add(r.choice(["Hasta Adı Soyadı: ", "Adı Soyadı: ", "Hasta: ", "Ad Soyad : "]))
    d.add(name, "AD")
    d.add("\n")
    if r.random() < 0.9:
        d.add(r.choice(["T.C. Kimlik No: ", "TC No: ", "Kimlik: "]))
        d.add(rand_tckn(r), "TCKN")
        d.add("\n")
    if r.random() < 0.7:
        d.add("Doğum Tarihi: ")
        d.add(f"{r.randint(1, 28):02d}.{r.randint(1, 12):02d}.{r.randint(1940, 2010)}", "DOGUM_TARIHI")
        d.add("\n")
    if r.random() < 0.6:
        d.add(r.choice(["Telefon: ", "Tel: ", "İletişim: "]))
        d.add(rand_phone(r), "TELEFON")
        d.add("\n")
    if r.random() < 0.4:
        d.add("E-posta: ")
        local = name.split()[0].translate(str.maketrans("ÇĞİÖŞÜçğıöşü", "CGIOSUcgiosu")).lower()
        d.add(f"{local}.{r.randint(1, 99)}@example.com", "EMAIL")
        d.add("\n")
    if r.random() < 0.7:
        d.add(r.choice(["Protokol No: ", "Dosya No: "]))
        d.add(f"{r.randint(2020, 2026)}-{r.randint(100000, 999999)}", "KAYIT_NO")
        d.add("\n")
    if r.random() < 0.2:
        d.add("IBAN (iade için): ")
        d.add(rand_iban(r), "IBAN")
        d.add("\n")
    d.add(r.choice(["İsteyen Hekim: Dr. ", "Doktor: ", "Hekim: "]))
    d.add(doctor, "AD")
    d.add(f"\nNumune Tarihi: {r.randint(1, 28):02d}.{r.randint(1, 12):02d}.2026\n\n")
    d.add("TETKİK              SONUÇ    BİRİM     REFERANS\n")
    for test, unit, lo, hi in r.sample(LABS, r.randint(3, 7)):
        v = round(r.uniform(lo * 0.6 if lo else 0, hi * 1.4), 1)
        d.add(f"{test:<20}{v:<9}{unit:<10}{lo} - {hi}\n")
    # hard negatives
    if r.random() < 0.5:
        d.add(f"Barkod: {invalid_11(r)}\n")
    if r.random() < 0.3:
        d.add(f"Cihaz seri no: {r.randint(500, 559)}{r.randint(1000000, 9999999)}\n")
    # free-text mentions (repeat of the patient name, or a relative's name with no field label)
    if r.random() < 0.5:
        d.add("\nNot: ")
        d.add(name, "AD")
        d.add(" 3 ay sonra kontrole çağrıldı.\n")
    if r.random() < 0.3:
        rel = f"{r.choice(FIRST)} {r.choice(LAST)}"
        d.add("Refakatçi ")
        d.add(rel, "AD")
        d.add(" bilgilendirildi.\n")
    return d


def score(docs, ner=None):
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    errors = []
    for d in docs:
        pred = {(s.start, s.end, s.label) for s in _propagate(d.text, detect(d.text, ner))}
        gold = set(d.spans)
        for s in pred & gold:
            tp[s[2]] += 1
        for s in pred - gold:
            fp[s[2]] += 1
            errors.append(("FP", s[2], d.text[s[0]:s[1]]))
        for s in gold - pred:
            fn[s[2]] += 1
            errors.append(("FN", s[2], d.text[s[0]:s[1]]))
    rows = {}
    for lab in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0.0
        rc = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else 0.0
        rows[lab] = {"precision": round(p, 3), "recall": round(rc, 3),
                     "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0, "support": tp[lab] + fn[lab]}
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    P, R = T / (T + F), T / (T + N)
    rows["MICRO"] = {"precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / (P + R), 3),
                     "support": T + N}
    return rows, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ner", action="store_true", help="add the transformer NER backend")
    ap.add_argument("--out", default="results/pii_benchmark.json")
    a = ap.parse_args()
    r = random.Random(a.seed)
    docs = [make_report(r) for _ in range(a.n)]
    ner = None
    if a.ner:
        from scllm.pii.detectors import HFNer
        ner = HFNer()
    rows, errors = score(docs, ner)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n_reports": a.n, "seed": a.seed, "ner": a.ner, "scores": rows,
               "error_examples": errors[:40]}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{'label':<14}{'P':>7}{'R':>7}{'F1':>7}{'n':>7}")
    for lab, v in rows.items():
        print(f"{lab:<14}{v['precision']:>7.3f}{v['recall']:>7.3f}{v['f1']:>7.3f}{v['support']:>7}")


if __name__ == "__main__":
    main()
