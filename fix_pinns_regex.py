from pathlib import Path
from datetime import datetime

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8-sig")

marker = "    # Tangani pertanyaan konsep agar tidak masuk ke analisis dataset."
if source.count(marker) != 2:
    raise SystemExit(
        f"STOP: ditemukan {source.count(marker)} blok, bukan 2. "
        "File utama tidak diubah."
    )

backup = path.with_name(
    f"main_new.before_pinns_regex_fix_{datetime.now():%Y%m%d_%H%M%S}.py"
)

# Benahi regex yang ditulis dengan backslash berlebih.
source = source.replace(
    r'r"\\b(?:apa itu|jelaskan|definisi)\\s+pinns\\b"',
    r'r"\b(?:apa itu|jelaskan|definisi)\s+pinns\b"',
)

# Hapus blok kedua, mulai dari marker kedua hingga sebelum prediksi.
first = source.find(marker)
second = source.find(marker, first + len(marker))
prediction_marker = "    # Prioritaskan permintaan prediksi."
end = source.find(prediction_marker, second)

if second < 0 or end < 0:
    raise SystemExit("STOP: batas blok duplikat tidak ditemukan.")

source = source[:second] + source[end:]

if source.count(marker) != 1:
    raise SystemExit("STOP: validasi duplikasi gagal.")

if r'r"\b(?:apa itu|jelaskan|definisi)\s+pinns\b"' not in source:
    raise SystemExit("STOP: regex yang benar tidak ditemukan.")

try:
    compile(source, str(path), "exec")
except SyntaxError as exc:
    raise SystemExit(f"STOP: sintaks gagal pada baris {exc.lineno}.")

backup.write_text(path.read_text(encoding="utf-8-sig"), encoding="utf-8")
path.write_text(source, encoding="utf-8")

print("BERHASIL: regex diperbaiki dan blok duplikat dihapus.")
print(f"Backup: {backup}")
