"""Prompt construction. The model receives the masked report *and* the deterministic flags, so it explains
results instead of having to compute them (a cheap, effective guard against arithmetic hallucinations)."""
from __future__ import annotations

from typing import List

from .labs import LabResult

SYSTEM = (
    "Sen bir laboratuvar sonucu açıklama asistanısın. Görevin, verilen tahlil sonuçlarını hastanın anlayacağı "
    "sade bir Türkçeyle açıklamak.\n"
    "Kurallar:\n"
    "1. Sadece raporda verilen değerleri kullan; yeni sayı, test ya da tanı uydurma.\n"
    "2. Bir değerin düşük/yüksek/normal olduğunu raporun referans aralığına göre söyle; 'Hesaplanan durumlar' "
    "tablosu verilmişse ona uy.\n"
    "3. Kesin tanı koyma ve ilaç önerme; anormal sonuçlar için hekime danışılmasını öner.\n"
    "4. Köşeli parantezli yer tutucuları (ör. [AD_1]) olduğu gibi bırak.\n"
    "5. Önce anormal sonuçları, sonra normal olanları kısaca özetle."
)


def flags_table(labs: List[LabResult]) -> str:
    if not labs:
        return "(raporda referans aralıklı sonuç bulunamadı)"
    rows = [f"- {l.name}: {l.raw_value} {l.unit or ''} (referans {l.low:g}-{l.high:g}) → {l.status}".replace("  ", " ")
            for l in labs]
    return "\n".join(rows)


def build_messages(masked_report: str, labs: List[LabResult], include_flags: bool = True) -> list:
    """``include_flags=False`` is the ablation baseline: the model must read ranges and decide on its own."""
    flags = f"Hesaplanan durumlar:\n{flags_table(labs)}\n\n" if include_flags else ""
    user = f"Tahlil raporu:\n```\n{masked_report.strip()}\n```\n\n{flags}Bu sonuçları açıkla."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
