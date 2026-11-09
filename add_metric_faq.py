from pathlib import Path
import shutil

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8")

marker = "    # Prioritaskan permintaan prediksi."
if marker not in source:
    raise SystemExit("STOP: Titik penyisipan tidak ditemukan.")

if "FAQ_METRIK_EVALUASI_V1" in source:
    raise SystemExit("STOP: FAQ metrik sudah ada; tidak ada perubahan.")

faq_code = r'''
    # FAQ_METRIK_EVALUASI_V1
    # Penjelasan umum metrik evaluasi, bukan klaim hasil model aktual.
    metric_question = question.lower()

    metric_compare = re.search(
        r"\b(?:apa\s+perbedaan|perbedaan|bedanya|"
        r"bandingkan|beda)\b",
        metric_question,
    ) and re.search(
        r"\b(?:mae|rmse|r2|r²|r\s*2)\b",
        metric_question,
    )

    metric_match = re.search(
        r"\b(mae|rmse|r2|r²|r\s*2)\b",
        metric_question,
    )

    metric_explainer = re.search(
        r"\b(?:apa\s+itu|jelaskan|definisi|arti|"
        r"maksud|bagaimana\s+(?:cara\s+)?membaca|"
        r"cara\s+menginterpretasikan)\b",
        metric_question,
    )

    if metric_compare:
        metric_answer = (
            "MAE, RMSE, dan R² mengukur aspek kinerja model regresi "
            "yang berbeda. MAE adalah rata-rata nilai absolut selisih "
            "prediksi dan nilai aktual; satuannya sama dengan target. "
            "RMSE adalah akar dari rata-rata kuadrat selisih, sehingga "
            "lebih sensitif terhadap kesalahan besar; satuannya juga "
            "sama dengan target. R² menunjukkan seberapa baik model "
            "menjelaskan variasi target dibandingkan baseline rata-rata. "
            "R² yang mendekati 1 umumnya menunjukkan kecocokan yang "
            "lebih baik, tetapi R² dapat bernilai negatif pada data uji. "
            "MAE dan RMSE yang lebih rendah umumnya lebih baik. "
            "Bandingkan metrik pada data uji, target, dan pembagian data "
            "yang sama. Tidak ada ambang universal untuk menyatakan model "
            "bagus; pertimbangkan juga satuan target dan kebutuhan penggunaan."
        )
    elif metric_match and metric_explainer:
        metric_name = metric_match.group(1).replace(" ", "").upper()
        if metric_name == "R²":
            metric_name = "R²"
        elif metric_name in ("R2", "R 2"):
            metric_name = "R²"

        metric_definitions = {
            "MAE": (
                "MAE (Mean Absolute Error) adalah rata-rata nilai absolut "
                "selisih antara prediksi dan nilai aktual. Nilai MAE yang "
                "lebih rendah umumnya lebih baik, dan satuannya sama "
                "dengan target prediksi."
            ),
            "RMSE": (
                "RMSE (Root Mean Squared Error) adalah akar kuadrat dari "
                "rata-rata kuadrat selisih prediksi dan nilai aktual. "
                "RMSE memberi penalti lebih besar pada kesalahan besar. "
                "Nilai lebih rendah umumnya lebih baik, dan satuannya "
                "sama dengan target prediksi."
            ),
            "R²": (
                "R² (koefisien determinasi) menunjukkan seberapa baik model "
                "menjelaskan variasi nilai target dibandingkan baseline "
                "yang menggunakan rata-rata target. Nilai mendekati 1 "
                "umumnya menunjukkan kecocokan lebih baik, tetapi R² dapat "
                "bernilai negatif pada data uji. R² perlu dinilai bersama "
                "MAE atau RMSE dan konteks data."
            ),
        }
        metric_answer = metric_definitions[metric_name]
    else:
        metric_answer = None

    if metric_answer is not None:
        return {
            "success": True,
            "answer": metric_answer,
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

'''

backup = path.with_name("main_new.before_metric_faq.py")
shutil.copy2(path, backup)

insert_at = source.find(marker)
updated = source[:insert_at] + faq_code + source[insert_at:]

compile(updated, str(path), "exec")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: FAQ MAE, RMSE, dan R² ditambahkan.")
print(f"Backup: {backup}")
print("Pemeriksaan sintaks Python: LULUS")
