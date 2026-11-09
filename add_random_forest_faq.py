from pathlib import Path
from datetime import datetime

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8-sig")

marker = "    # Prioritaskan permintaan prediksi."

if source.count(marker) != 1:
    raise SystemExit("STOP: penanda tidak ditemukan secara unik. File tidak diubah.")

addition = r'''    # Jawaban pengetahuan umum tentang Random Forest.
    if re.search(
        r"\b(?:apa itu|jelaskan|definisi)\s+random\s+forest\b",
        question,
        re.IGNORECASE,
    ):
        return {
            "success": True,
            "answer": (
                "Random Forest adalah metode machine learning yang "
                "menggabungkan banyak pohon keputusan untuk menghasilkan "
                "prediksi. Untuk tugas regresi, prediksi akhir umumnya "
                "merupakan rata-rata prediksi dari pohon-pohon tersebut. "
                "Dalam analisis pertumbuhan pohon, metode ini dapat "
                "digunakan untuk memprediksi target seperti DBH, tinggi, "
                "atau volume berdasarkan fitur yang tersedia. "
                "Akurasi model tetap perlu dievaluasi menggunakan data "
                "pengujian yang sesuai."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

    if re.search(
        r"\b(?:apa perbedaan|bedanya|perbedaan)\s+"
        r"(?:antara\s+)?(?:pinns|physics-informed neural networks)"
        r"\s+(?:dan|dengan|vs\.?)\s+random\s+forest\b"
        r"|\b(?:apa perbedaan|bedanya|perbedaan)\s+"
        r"random\s+forest\s+(?:dan|dengan|vs\.?)\s+"
        r"(?:pinns|physics-informed neural networks)\b",
        question,
        re.IGNORECASE,
    ):
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

updated = source.replace(marker, addition + marker, 1)

try:
    compile(updated, str(path), "exec")
except SyntaxError as exc:
    raise SystemExit(
        f"STOP: sintaks gagal pada baris {exc.lineno}. File tidak diubah."
    )

backup = path.with_name(
    f"main_new.before_random_forest_faq_{datetime.now():%Y%m%d_%H%M%S}.py"
)
backup.write_text(source, encoding="utf-8")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: dua penanganan pertanyaan Random Forest ditambahkan.")
print(f"Backup: {backup}")
print("Validasi sintaks sebelum penyimpanan: LULUS")
