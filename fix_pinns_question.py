from pathlib import Path
from datetime import datetime

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8-sig")

marker = "    # Prioritaskan permintaan prediksi."

if source.count(marker) != 1:
    raise SystemExit(
        "STOP: penanda penanganan prediksi tidak ditemukan secara unik. "
        "File utama tidak diubah."
    )

addition = '''    # Tangani pertanyaan konsep agar tidak masuk ke analisis dataset.
    if re.search(
        r"\\\\b(?:apa itu|jelaskan|definisi)\\\\s+pinns\\\\b",
        question,
        re.IGNORECASE,
    ):
        return {
            "success": True,
            "answer": (
                "PINNs (Physics-Informed Neural Networks) adalah jaringan "
                "saraf yang menggabungkan data dengan aturan atau persamaan "
                "fisika sebagai bagian dari proses pembelajaran. "
                "Untuk analisis pertumbuhan pohon, pendekatan ini dapat "
                "digunakan untuk mempelajari pola pertumbuhan dari data "
                "sekaligus mempertimbangkan hubungan matematis yang relevan. "
                "Kualitas hasilnya tetap bergantung pada data, persamaan "
                "yang digunakan, dan proses validasi model."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

'''

updated = source.replace(marker, addition + marker, 1)

try:
    compile(updated, str(path), "exec")
except SyntaxError as exc:
    raise SystemExit(
        f"STOP: validasi sintaks gagal pada baris {exc.lineno}. "
        "File utama tidak diubah."
    )

backup = path.with_name(
    f"main_new.before_pinns_question_{datetime.now():%Y%m%d_%H%M%S}.py"
)
backup.write_text(source, encoding="utf-8")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: penanganan pertanyaan PINNs ditambahkan.")
print(f"Backup: {backup}")
print("Validasi sintaks: LULUS")
