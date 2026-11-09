from pathlib import Path
import shutil

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8")

marker = "    prediction_result = answer_prediction_question(question)"
if marker not in source:
    raise SystemExit("STOP: Titik penyisipan tidak ditemukan.")

if "FAQ_ARSITEKTUR_DASHBOARD_V1" in source:
    raise SystemExit("STOP: FAQ arsitektur sudah terpasang.")

faq = r'''    # FAQ_ARSITEKTUR_DASHBOARD_V1
    architecture_question = question.lower()

    architecture_answers = [
        (
            r"\b(?:apa itu|jelaskan|fungsi)\s+frontend\b",
            "Frontend adalah bagian aplikasi yang berinteraksi langsung "
            "dengan pengguna, seperti halaman dashboard, formulir filter, "
            "tabel, dan grafik. Frontend mengirim permintaan ke backend "
            "ketika membutuhkan data atau hasil pemrosesan."
        ),
        (
            r"\b(?:apa itu|jelaskan|fungsi)\s+backend\b",
            "Backend adalah bagian aplikasi yang menangani logika "
            "pemrosesan, validasi permintaan, akses data, dan layanan "
            "yang dibutuhkan frontend. Pada proyek ini, backend "
            "menggunakan FastAPI berdasarkan kode api/main_new.py."
        ),
        (
            r"\b(?:apa itu|jelaskan|fungsi)\s+api\b",
            "API adalah antarmuka yang memungkinkan bagian aplikasi "
            "berkomunikasi melalui aturan permintaan dan respons. Dalam "
            "arsitektur frontend-backend, frontend dapat mengirim "
            "permintaan melalui API dan menerima data atau hasil "
            "pemrosesan dari backend."
        ),
        (
            r"\b(?:apa itu|jelaskan|fungsi)\s+database\b",
            "Database adalah sistem untuk menyimpan dan mengelola data "
            "agar dapat dicari, dibaca, ditambah, atau diperbarui. "
            "Dalam dashboard analitik, database dapat menjadi sumber "
            "data bagi backend. Jenis database dan cara koneksi yang "
            "benar-benar digunakan proyek perlu dikonfirmasi dari kode "
            "konfigurasi dan akses datanya."
        ),
    ]

    architecture_answer = None

    for architecture_pattern, architecture_text in architecture_answers:
        if re.search(architecture_pattern, architecture_question):
            architecture_answer = architecture_text
            break

    if architecture_answer is not None:
        return {
            "success": True,
            "answer": architecture_answer,
            "mode": "python_local",
            "data": {"found": False},
            "chart": None,
            "chart_offer": None,
        }

'''

backup = path.with_name("main_new.before_architecture_faq.py")
shutil.copy2(path, backup)

updated = source.replace(marker, faq + marker, 1)
compile(updated, str(path), "exec")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: FAQ arsitektur dasar ditambahkan.")
print(f"Backup: {backup}")
print("Pemeriksaan sintaks Python: LULUS")
