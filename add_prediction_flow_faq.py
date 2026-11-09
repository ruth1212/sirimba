from pathlib import Path
import shutil

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8")

marker = "    prediction_result = answer_prediction_question(question)"
if marker not in source:
    raise SystemExit("STOP: Titik penyisipan tidak ditemukan.")

if "FAQ_ALUR_PREDIKSI_V1" in source:
    raise SystemExit("STOP: FAQ alur prediksi sudah terpasang.")

faq = r'''    # FAQ_ALUR_PREDIKSI_V1
    prediction_flow_question = question.lower()

    prediction_flow_patterns = [
        (
            r"\b(?:bagaimana|jelaskan)\b.*"
            r"\b(?:alur|proses)\b.*\bprediksi\b",
            "Secara umum, alur prediksi dimulai ketika pengguna memilih "
            "parameter yang diperlukan. Frontend mengirim permintaan ke "
            "backend melalui API. Backend memvalidasi permintaan, "
            "menyiapkan input, memilih model yang sesuai, menjalankan "
            "prediksi, lalu mengirimkan hasil untuk ditampilkan frontend. "
            "Pada backend proyek ini tersedia fungsi predict_rf() untuk "
            "Random Forest dan predict_pinns() untuk PINNs. Detail "
            "permintaan dan respons setiap halaman bergantung pada kode "
            "frontend dan endpoint yang digunakan."
        ),
        (
            r"\b(?:apa fungsi|fungsi|jelaskan)\b.*"
            r"\b(?:predict_rf|random forest)\b",
            "Fungsi predict_rf() merupakan bagian backend yang menjalankan "
            "prediksi menggunakan model Random Forest. Berdasarkan kode "
            "yang tersedia, fungsi ini dipanggil ketika pemrosesan memilih "
            "model random_forest. Detail fitur input dan bentuk hasil "
            "prediksi mengikuti implementasi fungsi tersebut."
        ),
        (
            r"\b(?:apa fungsi|fungsi|jelaskan)\b.*"
            r"\b(?:predict_pinns|pinns)\b",
            "Fungsi predict_pinns() merupakan bagian backend yang "
            "menjalankan prediksi menggunakan model PINNs. Berdasarkan "
            "kode yang tersedia, fungsi ini dipanggil ketika pemrosesan "
            "memilih model pinns. Detail input, keluaran, dan komponen "
            "fisika perlu merujuk implementasi model yang sebenarnya."
        ),
        (
            r"\b(?:apa fungsi|fungsi|jelaskan)\b.*"
            r"\b(?:ai/chat|endpoint chat|chat ai)\b",
            "Endpoint POST /ai/chat menerima permintaan AI Growth "
            "Assistant melalui backend FastAPI. Handler ai_chat() "
            "memproses pertanyaan dan menentukan jenis jawaban yang "
            "sesuai, misalnya FAQ, analisis data, atau pertanyaan prediksi. "
            "Endpoint ini berbeda dari endpoint laporan PDF dan PPTX."
        ),
    ]

    prediction_flow_answer = None

    for flow_pattern, flow_answer in prediction_flow_patterns:
        if re.search(flow_pattern, prediction_flow_question):
            prediction_flow_answer = flow_answer
            break

    if prediction_flow_answer is not None:
        return {
            "success": True,
            "answer": prediction_flow_answer,
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

'''

backup = path.with_name("main_new.before_prediction_flow_faq.py")
shutil.copy2(path, backup)

updated = source.replace(marker, faq + marker, 1)
compile(updated, str(path), "exec")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: FAQ alur prediksi ditambahkan.")
print(f"Backup: {backup}")
print("Pemeriksaan sintaks Python: LULUS")
