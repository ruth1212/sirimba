from pathlib import Path
import shutil

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8")

marker = "    # FAQ_METRIK_EVALUASI_V1"

if marker not in source:
    raise SystemExit("STOP: Penanda FAQ metrik tidak ditemukan.")

if "FAQ_INTERPRETASI_R2_V1" in source:
    raise SystemExit("STOP: FAQ interpretasi R² sudah ada.")

faq = r'''
    # FAQ_INTERPRETASI_R2_V1
    r2_question = question.lower()

    r2_high_question = (
        re.search(r"\b(?:r\s*2|r²)\b", r2_question)
        and re.search(r"\b(?:tinggi|besar)\b", r2_question)
        and re.search(r"\b(?:bagus|baik|akurat|selalu)\b", r2_question)
    )

    if r2_high_question:
        return {
            "success": True,
            "answer": (
                "Tidak selalu. Nilai R² yang tinggi menunjukkan bahwa model "
                "dapat menjelaskan sebagian besar variasi target pada data "
                "yang dievaluasi, tetapi tidak otomatis berarti prediksinya "
                "akurat untuk semua kebutuhan. Periksa juga MAE dan RMSE "
                "untuk mengetahui besar kesalahan prediksi, pastikan evaluasi "
                "dilakukan pada data uji yang sesuai, dan waspadai kebocoran "
                "data atau pembagian data yang tidak mewakili penggunaan nyata. "
                "Untuk pertumbuhan pohon, periksa metrik pada target dan "
                "satuan masing-masing, misalnya DBH, tinggi, atau volume. "
                "Kesimpulan tentang kualitas model harus didasarkan pada "
                "hasil evaluasi aktual, bukan R² saja."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

'''

backup = path.with_name("main_new.before_r2_interpretation_faq.py")
shutil.copy2(path, backup)

insert_at = source.find(marker)
updated = source[:insert_at] + faq + source[insert_at:]

compile(updated, str(path), "exec")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: FAQ interpretasi R² ditambahkan.")
print(f"Backup: {backup}")
print("Pemeriksaan sintaks Python: LULUS")
