from pathlib import Path
import re
import shutil

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8")

start_marker = "    # Normalisasi ringan untuk FAQ perbandingan metode."
end_marker = "    # Prioritaskan permintaan prediksi."

start = source.find(start_marker)
end = source.find(end_marker, start)

if start == -1 or end == -1:
    raise SystemExit("STOP: Blok FAQ tidak ditemukan. File tidak diubah.")

replacement = r'''    # Normalisasi ringan untuk FAQ perbandingan metode.
    comparison_faq_text = question.lower()
    comparison_faq_text = re.sub(
        r"\brndom\b", "random", comparison_faq_text
    )
    comparison_faq_text = re.sub(
        r"\bperbeda+a+n\b", "perbedaan", comparison_faq_text
    )

    comparison_pattern = (
        r"\b(?:apa\s+(?:itu\s+)?(?:perbedaan|bedanya)|"
        r"bedanya|perbedaan)\s+"
        r"(?:antara\s+)?"
        r"(?:pinns|physics-informed neural networks)"
        r"\s+(?:dan|dengan|vs\.?)\s+random\s+forest\b"
        r"|\b(?:apa\s+(?:itu\s+)?(?:perbedaan|bedanya)|"
        r"bedanya|perbedaan)\s+"
        r"random\s+forest\s+(?:dan|dengan|vs\.?)\s+"
        r"(?:pinns|physics-informed neural networks)\b"
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

'''

backup = path.with_name("main_new.before_regex_repair.py")
shutil.copy2(path, backup)

updated = source[:start] + replacement + source[end:]
compile(updated, str(path), "exec")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: Regex FAQ diperbaiki.")
print(f"Backup tersimpan: {backup}")
print("Pemeriksaan sintaks Python: LULUS")
'''

Path("fix_comparison_regex.py").write_text(replacement, encoding="utf-8")
