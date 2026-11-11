import pytest

from scllm.pii.detectors import detect, iban_valid, tckn_valid
from scllm.pii.masker import mask, unmask

REPORT = """HASTA BİLGİLERİ
Hasta Adı Soyadı: Ayşe Yılmaz
T.C. Kimlik No: 10000000146
Doğum Tarihi: 12.03.1985
Telefon: 0532 123 45 67
E-posta: ayse.yilmaz@example.com
Protokol No: 2026-118734
İsteyen Hekim: Dr. Mehmet Demir

TETKİK              SONUÇ    BİRİM     REFERANS
Hemoglobin          10.2     g/dL      12.0 - 16.0
Glukoz (açlık)      126      mg/dL     70 - 100
Barkod 48213907715 numaralı örnek.
"""


def test_tckn_checksum():
    assert tckn_valid("10000000146")
    assert not tckn_valid("10000000147") and not tckn_valid("01234567890")


def test_iban_checksum():
    assert iban_valid("TR33 0006 1005 1978 6457 8413 26")
    assert not iban_valid("TR34 0006 1005 1978 6457 8413 26")


def test_report_fields_are_found_and_lab_values_kept():
    labels = {(s.label, s.text) for s in detect(REPORT)}
    assert ("AD", "Ayşe Yılmaz") in labels
    assert ("AD", "Mehmet Demir") in labels
    assert ("TCKN", "10000000146") in labels
    assert ("DOGUM_TARIHI", "12.03.1985") in labels
    assert ("EMAIL", "ayse.yilmaz@example.com") in labels
    assert ("KAYIT_NO", "2026-118734") in labels
    assert any(l == "TELEFON" for l, _ in labels)
    texts = {t for _, t in labels}
    assert not any(v in texts for v in ("10.2", "126", "48213907715"))  # invalid TCKN barcode is not PII


def test_mask_unmask_round_trip_and_consistency():
    r = mask(REPORT + "\nAyşe Yılmaz kontrole gelecek.")
    assert "Ayşe" not in r.text and "10000000146" not in r.text
    assert r.text.count("[AD_1]") == 2  # header value propagates to the free-text mention
    assert "Hemoglobin          10.2" in r.text
    answer = "[AD_1] için hemoglobin düşük. [TCKN_1] numaralı hasta. [AD_9] bilinmiyor."
    restored = unmask(answer, r.mapping)
    assert restored.startswith("Ayşe Yılmaz için") and "10000000146" in restored and "[AD_9]" in restored


def test_repeated_value_gets_same_placeholder():
    r = mask("Telefon: 05321234567\nAcil telefon: 05321234567")
    assert r.text.count("[TELEFON_1]") == 2 and len(r.mapping) == 1
