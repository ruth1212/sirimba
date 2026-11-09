from pathlib import Path

from api.report_generator import (
    generate_pdf_report,
    generate_pptx_report,
)

sample_result = {
    "success": True,
    "answer": (
        "Hasil pengujian contoh. "
        "Kategori A memiliki nilai tertinggi pada data simulasi."
    ),
    "data": {
        "metric": "height_m",
        "metric_label": "Tinggi pohon (m)",
        "group_by": "regional_name",
        "group_label": "Regional",
        "period": "P36",
        "groups": 3,
        "filters": {
            "regional": None,
            "species": None,
            "compartments": [],
            "clone": None,
        },
    },
    "chart": {
        "type": "bar",
        "title": "Uji Perbandingan Tinggi Pohon",
        "x_label": "Regional",
        "y_label": "Tinggi pohon (m)",
        "period": "P36",
        "data": [
            {"category": "Kategori A", "value": 23.5},
            {"category": "Kategori B", "value": 21.2},
            {"category": "Kategori C", "value": 19.8},
        ],
    },
}

output_dir = Path("results") / "reports_test"
output_dir.mkdir(parents=True, exist_ok=True)

pdf_buffer = generate_pdf_report(
    sample_result,
    "Uji perbandingan tinggi pohon periode P36",
)
pdf_path = output_dir / "test_report.pdf"
pdf_path.write_bytes(pdf_buffer.getvalue())

pptx_buffer = generate_pptx_report(
    sample_result,
    "Uji perbandingan tinggi pohon periode P36",
)
pptx_path = output_dir / "test_report.pptx"
pptx_path.write_bytes(pptx_buffer.getvalue())

print("PDF:", pdf_path.resolve())
print("PDF size:", pdf_path.stat().st_size, "bytes")
print("PPTX:", pptx_path.resolve())
print("PPTX size:", pptx_path.stat().st_size, "bytes")

if pdf_path.stat().st_size <= 0 or pptx_path.stat().st_size <= 0:
    raise RuntimeError("File laporan kosong.")

print("STATUS: KEDUA FILE BERHASIL DIBUAT")
print("CATATAN: Isi file ini menggunakan DATA SIMULASI.")
