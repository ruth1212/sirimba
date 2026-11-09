from pathlib import Path
import shutil

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8")

marker = "    prediction_result = answer_prediction_question(question)"
if marker not in source:
    raise SystemExit("STOP: Handler prediksi tidak ditemukan.")

if "FAQ_KUALITAS_MODEL_V1" in source:
    raise SystemExit("STOP: FAQ kualitas model sudah terpasang.")

faq = r'''    # FAQ_KUALITAS_MODEL_V1
    # Pertanyaan evaluasi umum harus dijawab sebelum handler prediksi.
    model_quality_question = question.lower()

    asks_model_quality = (
        ("bagaimana" in model_quality_question
         and any(word in model_quality_question for word in
                 ("bagus", "baik", "akurat", "kualitas")))
        or any(phrase in model_quality_question for phrase in (
            "menilai model",
            "evaluasi model",
            "kualitas model",
            "model yang bagus",
            "model yang baik",
            "prediksi bagus atau tidak",
            "prediksi baik atau tidak",
            "mengukur akurasi model",
        ))
    )

    asks_for_actual_prediction = any(phrase in model_quality_question for phrase in (
        "prediksi tinggi",
        "prediksi diameter",
        "prediksi dbh",
        "prediksi volume",
        "berapa tinggi",
        "berapa diameter",
        "berapa volume",
        "hitung prediksi",
        "buat prediksi",
    ))

    if asks_model_quality and not asks_for_actual_prediction:
        return {
            "success": True,
            "answer": (
                "Kualitas model prediksi dinilai menggunakan data uji "
                "yang tidak digunakan saat pelatihan. Untuk regresi, "
                "periksa MAE dan RMSE—semakin rendah umumnya semakin baik—"
                "serta R², yang umumnya semakin baik jika mendekati 1. "
                "Nilai tersebut harus dibaca bersama satuan dan rentang "
                "target, misalnya DBH (cm), tinggi (m), dan volume (m³ "
                "per pohon). Pastikan pembagian data mewakili kondisi "
                "penggunaan nyata dan tidak mengalami kebocoran data. "
                "Bandingkan model pada data uji, target, dan pembagian "
                "data yang sama. Saya belum menyimpulkan kualitas model "
                "aktual karena pertanyaan ini belum menyertakan hasil "
                "evaluasi MAE, RMSE, dan R² yang relevan."
            ),
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

'''

backup = path.with_name("main_new.before_model_quality_faq.py")
shutil.copy2(path, backup)

updated = source.replace(marker, faq + marker, 1)
compile(updated, str(path), "exec")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: FAQ kualitas model ditambahkan.")
print(f"Backup: {backup}")
print("Pemeriksaan sintaks Python: LULUS")
