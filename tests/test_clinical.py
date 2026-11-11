from scllm.clinical.grounding import check
from scllm.clinical.labs import parse_labs
from scllm.clinical.prompt import build_messages

REPORT = """TETKİK              SONUÇ    BİRİM     REFERANS
Hemoglobin          10,2     g/dL      12.0 - 16.0
Glukoz (açlık)      126      mg/dL     70 - 100   H
TSH: 2.1 mIU/L (0.4-4.0)
Ferritin            8        ng/mL     15 - 150
Not: 3 gün sonra tekrar.
"""


def test_parse_and_flag():
    labs = {l.name: l for l in parse_labs(REPORT)}
    assert set(labs) == {"Hemoglobin", "Glukoz (açlık)", "TSH", "Ferritin"}
    assert labs["Hemoglobin"].value == 10.2 and labs["Hemoglobin"].status == "düşük"
    assert labs["Glukoz (açlık)"].status == "yüksek" and labs["TSH"].status == "normal"
    assert labs["Ferritin"].deviation < 0


def test_prompt_contains_flags_and_rules():
    msgs = build_messages(REPORT, parse_labs(REPORT))
    assert msgs[0]["role"] == "system" and "uydurma" in msgs[0]["content"]
    assert "Hemoglobin: 10,2 g/dL (referans 12-16) → düşük" in msgs[1]["content"]


def test_grounding_catches_invented_numbers_and_wrong_direction():
    labs = parse_labs(REPORT)
    good = "Hemoglobin değeriniz 10,2 ile düşük. TSH normal. [AD_1] için ferritin düşük."
    assert check(good, REPORT, labs).ok
    bad = "Hemoglobin değeriniz yüksek. Kolesterol 240 çıkmış."
    r = check(bad, REPORT, labs)
    assert r.unsupported_numbers == ["240"]
    assert r.contradictions and r.contradictions[0]["test"] == "Hemoglobin"
