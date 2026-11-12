from scllm.clinical.labs import parse_labs
from scllm.clinical.structured import parse_structured, render
from scllm.pii.gate import PrivacyGate
from scllm.pii.masker import mask
from scllm.pii.synthetic import generate, name_pools, suffix

SRC = "Ferritin  180.5  ng/mL  15 - 150\nTSH  2.8  mIU/L  0.4 - 4.0\n"


def test_lenient_json_parsing():
    raw = 'Tabii! ```json\n{"ozet": "x", "testler": [{"test": "TSH", "aciklama": "Tiroid {hormonu}."},], "oneri": ""}\n```'
    p = parse_structured(raw)
    assert p["testler"] == [{"test": "TSH", "aciklama": "Tiroid {hormonu}."}]
    assert parse_structured("JSON yok") is None


def test_render_falls_back_to_computed_table():
    labs = parse_labs(SRC)
    a = render("bozuk çıktı", SRC, labs)
    assert not a.parsed and "Ferritin: 180.5 ng/mL** (referans 15–150) → Yüksek" in a.text
    assert a.text.index("Ferritin") < a.text.index("TSH")  # abnormal first


def test_render_drops_invented_numbers_and_wrong_directions():
    labs = parse_labs(SRC)
    raw = ('{"ozet": "", "testler": [{"test": "Ferritin", "aciklama": "Demir deposunu gösterir. Değer 400 civarında."},'
           ' {"test": "TSH (tiroid)", "aciklama": "Tiroid hormonu düşük görünüyor."}], "oneri": ""}')
    a = render(raw, SRC, labs)
    assert "400" not in a.text and "düşük görünüyor" not in a.text and "Demir deposunu gösterir." in a.text
    assert len(a.removed_sentences) == 2


def test_gate_passes_clean_and_blocks_leftovers():
    clean = mask("Hasta Adı Soyadı: Ayşe Yılmaz\nTC: 10000000146\n" + SRC)
    assert PrivacyGate().check(clean).ok
    leftover = mask("Kontrol için 5321234567 aranacak.\n" + SRC)  # bare number, no phone keyword: not masked
    rep = PrivacyGate().check(leftover)
    assert not rep.ok and rep.to_dict()["findings"][0]["label"] == "NUMBER"
    assert "5321234567" not in str(rep.to_dict())  # values never reach logs


def test_synthetic_splits_are_disjoint():
    (f1, l1), (f2, l2) = name_pools("train"), name_pools("test")
    assert not set(f1) & set(f2) and not set(l1) & set(l2)
    assert {d.family for d in generate(40, "train", 1)}.isdisjoint({d.family for d in generate(40, "test", 1)})
    for d in generate(30, "test", 2):
        for a, b, lab in d.spans:
            assert d.text[a:b].strip() == d.text[a:b] and b > a


def test_vowel_harmony():
    assert [suffix(w, "gen") for w in ("Kaya", "Demir", "Koç", "Öztürk")] == ["'nın", "'in", "'un", "'ün"]
    assert suffix("Koç", "abl") == "'tan" and suffix("Kaya", "dat") == "'ya"


def test_truncated_json_keeps_completed_tests():
    raw = '{"ozet": "Kısa özet.", "testler": [{"test": "TSH", "aciklama": "Tiroid hormonu."}, {"test": "Ferritin", "aci'
    p = parse_structured(raw)
    assert p["ozet"] == "Kısa özet." and p["testler"] == [{"test": "TSH", "aciklama": "Tiroid hormonu."}]


def test_echoed_flag_line_is_not_shown_twice():
    labs = parse_labs(SRC)
    raw = '{"ozet": "", "testler": [{"test": "Ferritin", "aciklama": "180.5 ng/mL (referans 15-150) → yüksek"}], "oneri": ""}'
    a = render(raw, SRC, labs)
    assert a.text.count("180.5") == 1 and a.explained == 0 and not a.removed_sentences
