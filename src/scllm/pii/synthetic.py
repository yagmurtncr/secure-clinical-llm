"""Synthetic Turkish clinical documents with gold PII spans, split into *template families* and *name pools*.

Train and test never share a family or a name, so a detector tuned or fine-tuned on the train split is measured
on phrasing and names it has not seen. Documents mix structured headers (lab reports) with free clinical text
(discharge summaries, consult notes, referral letters) where names appear with Turkish case suffixes
("Ayşe Yılmaz'ın"), in upper / lower case, with titles, and as relatives or clinicians.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

FIRST = ["Ayşe", "Mehmet", "Fatma", "Mustafa", "Emine", "Ahmet", "Hatice", "Ali", "Zeynep", "Hüseyin", "Elif",
         "Hasan", "Merve", "İbrahim", "Şule", "Can", "Özge", "Murat", "Gül", "Ömer", "Çağla", "Yusuf", "Büşra",
         "Emre", "Ece", "Burak", "Derya", "Kerem", "İrem", "Serkan", "Sevgi", "Halil", "Esra", "Osman", "Tuğba",
         "Kemal", "Dilek", "Recep", "Gamze", "Erkan", "Sibel", "Volkan", "Pınar", "Cem", "Aslı", "Tolga", "Melek",
         "Barış", "Nazlı", "Okan", "Sinem", "Uğur", "Ebru", "Levent", "Filiz", "Onur", "Hande", "Sinan", "Yasemin",
         "Orhan", "Gizem", "Tarık", "Nilay", "Bülent", "Seda", "Hakan", "Duygu", "İsmail", "Öykü", "Ferhat",
         "Bahar", "Kadir", "Ceren", "Selim", "Damla", "Yavuz", "Leyla", "Arda", "Nesrin", "Eren", "Gülsüm", "Taner",
         "Aysel", "Rıza", "Şeyma", "Cengiz", "Hülya", "Metin", "Belgin", "Ercan", "Yıldız", "Ozan", "Songül",
         "Sami", "Elçin", "Doğan", "Nermin", "Fikret", "İlknur", "Necati"]
LAST = ["Yılmaz", "Kaya", "Demir", "Şahin", "Çelik", "Yıldız", "Yıldırım", "Öztürk", "Aydın", "Özdemir",
        "Arslan", "Doğan", "Kılıç", "Aslan", "Çetin", "Kara", "Koç", "Kurt", "Özkan", "Şimşek", "Güneş",
        "Erdoğan", "Polat", "Korkmaz", "Avcı", "Taş", "Aksoy", "Kaplan", "Bulut", "Keskin", "Ünal", "Acar",
        "Tekin", "Bozkurt", "Uçar", "Turan", "Sarı", "Er", "Güler", "Karaca", "Işık", "Tunç", "Coşkun", "Altun",
        "Gündoğdu", "Bayram", "Yavuz", "Akın", "Uysal", "Toprak", "Ateş", "Erol", "Kocaman", "Sezer", "Durmaz",
        "Kahraman", "Çakır", "Özer", "Tuncer", "Duman", "Karakaya", "Başaran", "Eren", "Yücel", "Akgül",
        "Gök", "Oral", "Bilgin", "Köse", "Demirci", "Önal", "Ekinci", "Çiftçi", "Sönmez", "Kalkan", "Ay"]


def name_pools(split: str) -> Tuple[List[str], List[str]]:
    """Disjoint halves of the name lists: 'train' gets even indices, 'test' odd ones."""
    k = 0 if split == "train" else 1
    return FIRST[k::2], LAST[k::2]


LABS = [("Hemoglobin", "g/dL", 12.0, 16.0), ("Glukoz (açlık)", "mg/dL", 70, 100), ("TSH", "mIU/L", 0.4, 4.0),
        ("Ferritin", "ng/mL", 15, 150), ("ALT", "U/L", 0, 41), ("Kreatinin", "mg/dL", 0.6, 1.2),
        ("LDL Kolesterol", "mg/dL", 0, 130), ("B12 Vitamini", "pg/mL", 200, 900), ("WBC", "10^3/µL", 4.0, 10.0),
        ("CRP", "mg/L", 0, 5), ("HbA1c", "%", 4.0, 5.6), ("Sodyum", "mmol/L", 135, 145)]


def rand_tckn(r: random.Random) -> str:
    d = [r.randint(1, 9)] + [r.randint(0, 9) for _ in range(8)]
    d.append(((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10)
    d.append(sum(d) % 10)
    return "".join(map(str, d))


def rand_phone(r: random.Random) -> str:
    a, b, c, d = r.randint(500, 559), r.randint(100, 999), r.randint(10, 99), r.randint(10, 99)
    return r.choice([f"0{a} {b} {c} {d}", f"+90 {a} {b} {c} {d}", f"0{a}{b}{c}{d}", f"0{a}-{b}-{c}-{d}"])


@dataclass
class Doc:
    parts: List[str] = field(default_factory=list)
    spans: List[Tuple[int, int, str]] = field(default_factory=list)
    family: str = ""

    def add(self, s: str, label: str = "") -> "Doc":
        start = sum(len(p) for p in self.parts)
        self.parts.append(s)
        if label:
            self.spans.append((start, start + len(s), label))
        return self

    @property
    def text(self) -> str:
        return "".join(self.parts)


_BACK = "aıou"
_VOWELS = "aeıioöuü"


def suffix(word: str, case: str) -> str:
    """Turkish case ending with vowel harmony: genitive ('ın/'nin/'un…), dative ('a/'ye…), ablative ('dan/'ten…)."""
    w = word.lower()
    last_v = next((ch for ch in reversed(w) if ch in _VOWELS), "e")
    ends_vowel = w[-1] in _VOWELS
    if case == "gen":
        v = {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u", "ö": "ü", "ü": "ü"}[last_v]
        return "'" + ("n" if ends_vowel else "") + v + "n"
    two = "a" if last_v in _BACK else "e"
    if case == "dat":
        return "'" + ("y" if ends_vowel else "") + two
    hard = w[-1] in "çfhkpsşt"
    return "'" + ("t" if hard else "d") + two + "n"


class Ctx:
    def __init__(self, r: random.Random, split: str):
        self.r = r
        self.first, self.last = name_pools(split)

    def person(self, parts: int = 0) -> str:
        r = self.r
        n = parts or r.choices([2, 3], [0.85, 0.15])[0]
        firsts = r.sample(self.first, n - 1)
        return " ".join(firsts + [r.choice(self.last)])

    def cased(self, name: str) -> str:
        style = self.r.choices(["title", "upper", "lower"], [0.75, 0.2, 0.05])[0]
        if style == "upper":
            return name.replace("i", "İ").replace("ı", "I").upper()
        if style == "lower":
            return name.replace("İ", "i").replace("I", "ı").lower()
        return name

    def labs(self, d: Doc, k=(3, 6)):
        d.add("\nTETKİK              SONUÇ    BİRİM     REFERANS\n")
        for test, unit, lo, hi in self.r.sample(LABS, self.r.randint(*k)):
            v = round(self.r.uniform(lo * 0.6 if lo else 0, hi * 1.4), 1)
            d.add(f"{test:<20}{v:<9}{unit:<10}{lo} - {hi}\n")

    def age(self) -> int:
        return self.r.randint(18, 90)


# ---------------------------------------------------------------- train families


def t_lab_header(c: Ctx) -> Doc:
    d, r = Doc(), c.r
    p, dr = c.person(), c.person(2)
    d.add(r.choice(["LABORATUVAR SONUÇ RAPORU\n", "BİYOKİMYA\n", ""]))
    d.add(r.choice(["Hasta Adı Soyadı: ", "Adı Soyadı: ", "Hasta: "])).add(c.cased(p), "AD").add("\n")
    d.add("T.C. Kimlik No: ").add(rand_tckn(r), "TCKN").add("\n")
    if r.random() < 0.5:
        d.add("Tel: ").add(rand_phone(r), "TELEFON").add("\n")
    d.add(r.choice(["İsteyen Hekim: Dr. ", "Hekim: "])).add(dr, "AD").add("\n")
    c.labs(d)
    return d


def t_discharge(c: Ctx) -> Doc:
    d, r = Doc(), c.r
    p, dr = c.person(), c.person(2)
    d.add("EPİKRİZ\n\n")
    d.add(f"{c.age()} yaşında {r.choice(['kadın', 'erkek'])} hasta ").add(c.cased(p), "AD")
    d.add(r.choice([" nefes darlığı şikayetiyle acil servise başvurdu. ",
                    " göğüs ağrısı nedeniyle yatırıldı. ", " ateş ve halsizlik ile kliniğimize kabul edildi. "]))
    d.add("Yatışı süresince ").add(p, "AD").add(suffix(p, "gen"))
    d.add(" vital bulguları stabil seyretti.")
    if r.random() < 0.6:
        rel = c.person(2)
        d.add(f" Hasta yakını ({r.choice(['kızı', 'oğlu', 'eşi'])}) ").add(rel, "AD")
        d.add(" taburculuk sonrası bakım konusunda bilgilendirildi.")
    d.add("\nSorumlu hekim: Uzm. Dr. ").add(dr, "AD").add("\n")
    c.labs(d, (2, 4))
    return d


def t_consult(c: Ctx) -> Doc:
    d, r = Doc(), c.r
    p, dr = c.person(), c.person(2)
    d.add("KONSÜLTASYON NOTU\n")
    d.add("İstem: Dr. ").add(dr, "AD").add("\n")
    d.add("Hasta ").add(p, "AD").add(f" ({c.age()} y) ")
    d.add(r.choice(["diyabet takibi için", "anemi etiyolojisi açısından", "tiroid fonksiyonları için"]))
    d.add(" değerlendirildi. TC: ").add(rand_tckn(r), "TCKN")
    d.add(". Ek tetkik önerildi; sonuçlar ").add(p.split()[0], "AD")
    d.add(r.choice([" Hanım", " Bey"])).add(" ile paylaşılacak.\n")
    c.labs(d, (2, 4))
    return d


def t_referral(c: Ctx) -> Doc:
    d, r = Doc(), c.r
    p, dr, dr2 = c.person(), c.person(2), c.person(2)
    d.add("Sayın Dr. ").add(dr2, "AD").add(",\n\n")
    d.add("Kliniğimizde takip edilen ").add(c.cased(p), "AD")
    d.add(f" ({c.age()} yaş) {r.choice(['nefroloji', 'endokrinoloji', 'hematoloji'])} değerlendirmesi için ")
    d.add("tarafınıza yönlendirilmiştir. Ekteki sonuçlar ").add(p, "AD").add(suffix(p, "abl"))
    d.add(" alınan kan örneğine aittir. İletişim: ").add(rand_phone(r), "TELEFON").add(".\n\nSaygılarımla,\nDr. ")
    d.add(dr, "AD").add("\n")
    c.labs(d, (2, 4))
    return d


# ---------------------------------------------------------------- held-out families (never used for training)


def h_nursing(c: Ctx) -> Doc:
    d, r = Doc(), c.r
    p, nurse = c.person(), c.person(2)
    d.add("HEMŞİRE GÖZLEM FORMU\n")
    d.add(f"Saat {r.randint(0, 23):02d}:{r.choice(['00', '15', '30', '45'])} — ")
    shown = c.cased(p)
    d.add(shown, "AD").add(suffix(p, "gen"))
    d.add(" ateşi ölçüldü, ağrı skoru 3/10. ")
    d.add("Refakatçisi ").add(c.person(2), "AD").add(" odada.\n")
    d.add("Gözlemi yapan: Hemş. ").add(nurse, "AD").add("\n")
    c.labs(d, (2, 3))
    return d


def h_phone_log(c: Ctx) -> Doc:
    d, r = Doc(), c.r
    p, dr = c.person(), c.person(2)
    d.add("Telefon görüşmesi kaydı: ")
    d.add(dr, "AD").add(" hastanın kızı ").add(c.person(2), "AD")
    d.add(" ile görüştü (").add(rand_phone(r), "TELEFON").add("). ")
    d.add("Hasta ").add(c.cased(p), "AD").add(" yarın sabah kontrole gelecek; kimlik numarası ")
    tc = rand_tckn(r)
    d.add(f"{tc[:3]} {tc[3:6]} {tc[6:9]} {tc[9:]}", "TCKN").add(".\n")
    c.labs(d, (2, 3))
    return d


def h_table_row(c: Ctx) -> Doc:
    d, r = Doc(), c.r
    d.add("NÖBET DEVİR LİSTESİ\nYatak | Hasta | Not\n")
    for bed in r.sample(range(1, 30), r.randint(2, 4)):
        d.add(f"{bed:>4} | ").add(c.cased(c.person()), "AD").add(f" | {r.choice(['NPO', 'mobilize', 'O2 2 L/dk', 'izlem'])}\n")
    d.add("Devreden: ").add(c.person(2), "AD").add("  Devralan: ").add(c.person(2), "AD").add("\n")
    c.labs(d, (2, 3))
    return d


TRAIN_FAMILIES: Dict[str, Callable[[Ctx], Doc]] = {"lab_header": t_lab_header, "discharge": t_discharge,
                                                     "consult": t_consult, "referral": t_referral}
TEST_FAMILIES: Dict[str, Callable[[Ctx], Doc]] = {"nursing_note": h_nursing, "phone_log": h_phone_log,
                                                    "handover_table": h_table_row}


def generate(n: int, split: str, seed: int = 0) -> List[Doc]:
    """``split='train'`` uses train families and train names; ``'test'`` uses held-out families and test names;
    ``'test_seen_templates'`` uses train families with test names (isolates the template effect)."""
    r = random.Random(seed)
    fams = TEST_FAMILIES if split == "test" else TRAIN_FAMILIES
    c = Ctx(r, "train" if split == "train" else "test")
    out = []
    for _ in range(n):
        name, fn = r.choice(sorted(fams.items()))
        d = fn(c)
        d.family = name
        out.append(d)
    return out
