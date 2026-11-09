from pathlib import Path
import re
import shutil

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8")

start_marker = '    if re.search(\n        r"\\b(?:apa perbedaan|bedanya|perbedaan)\\s+"'
start = source.find(start_marker)

if start == -1:
    raise SystemExit(
        "STOP: Blok FAQ perbandingan tidak ditemukan. "
        "File tidak diubah."
    )

end_marker = '        } \n\n    # Prioritaskan permintaan prediksi.'
end = source.find(end_marker, start)

if end == -1:
    end_marker = '        }\n\n    # Prioritaskan permintaan prediksi.'
    end = source.find(end_marker, start)

if end == -1:
    raise SystemExit(
        "STOP: Batas akhir blok FAQ tidak ditemukan. "
        "File tidak diubah."
    )

end += len(end_marker.split("\n\n    # Prioritaskan")[0])

replacement = '''    # Normalisasi ringan untuk FAQ perbandingan metode.
    comparison_faq_text = question.lower()
    comparison_faq_text = re.sub(
        r"\\\\brndom\\\\b", "random", comparison_faq_text
    )
    comparison_faq_text = re.sub(
        r"\\\\bperbeda+a+n\\\\b", "perbedaan", comparison_faq_text
    )

    comparison_pattern = (
        r"\\\\b(?:apa\\\\s+(?:itu\\\\s+)?(?:perbedaan|bedanya)|"
        r"bedanya|perbedaan)\\\\s+"
        r"(?:antara\\\\s+)?"
        r"(?:pinns|physics-informed neural networks)"
        r"\\\\s+(?:dan|dengan|vs\\\\.?)\\\\s+random\\\\s+forest\\\\b"
        r"|\\\\b(?:apa\\\\s+(?:itu\\\\s+)?(?:perbedaan|bedanya)|"
        r"bedanya|perbedaan)\\\\s+"
        r"random\\\\s+forest\\\\s+(?:dan|dengan|vs\\\\.?)\\\\s+"
        r"(?:pinns|physics-informed neural networks)\\\\b"
    )

    if re.search(comparison_pattern, comparison_faq_text, re.IGNORECASE):
        return {
            "success": True,
            "answer": (
                "PINNs dan Random Forest memiliki pendekatan berbeda. "
                "Random Forest menggabungkan prediksi dari banyak pohon "
                "keputusan. PINNs menggunakan jaringan saraf dan dapat "
                "memasukkan persamaan atau batasan fisika ke dalam proses "
                "pelatihan melalui komponen physics loss. "
                "Random Forest tidak secara inheren menerapkan batasan "
                "fisika tersebut. Pada dashboard pertumbuhan pohon, "
                "perbandingan kinerja kedua metode harus didasarkan pada "
                "data pengujian, target prediksi, dan metrik evaluasi "
                "yang sama. Tidak ada metode yang otomatis selalu lebih baik."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    # Prioritaskan permintaan prediksi.'''

backup = path.with_name("main_new.before_faq_comparison_fix.py")
shutil.copy2(path, backup)

updated = source[:start] + replacement + source[end:]
compile(updated, str(path), "exec")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: FAQ perbandingan diperbaiki.")
print(f"Backup: {backup}")
print("Pemeriksaan sintaks Python: LULUS")
