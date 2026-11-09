
from datetime import datetime
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


# ============================================================
# TREE GROWTH ANALYTICS - REPORT GENERATOR
# ============================================================

REPORT_TITLE = "TREE GROWTH ANALYTICS"
REPORT_SUBTITLE = "Laporan Analisis Perbandingan Pertumbuhan Pohon"


def _validate_result(result: dict) -> None:
    """Pastikan hasil analisis cukup lengkap untuk dibuatkan laporan."""

    if not isinstance(result, dict):
        raise ValueError("Hasil analisis harus berupa dictionary.")

    if result.get("success") is not True:
        raise ValueError("Hasil analisis tidak menyatakan success=True.")

    data = result.get("data")
    chart = result.get("chart")

    if not isinstance(data, dict):
        raise ValueError("Bagian data analisis tidak tersedia.")

    if not isinstance(chart, dict):
        raise ValueError(
            "Grafik perbandingan tidak tersedia. "
            "Gunakan pertanyaan perbandingan yang menghasilkan grafik."
        )

    chart_rows = chart.get("data")

    if not isinstance(chart_rows, list) or not chart_rows:
        raise ValueError("Data grafik kosong.")

    valid_rows = [
        row
        for row in chart_rows
        if isinstance(row, dict)
        and row.get("category") is not None
        and row.get("value") is not None
    ]

    if not valid_rows:
        raise ValueError("Tidak ditemukan kategori dan nilai grafik yang valid.")


def _get_report_data(result: dict, question: str) -> dict:
    """Normalisasi hasil analisis tanpa menghitung ulang dataset."""

    _validate_result(result)

    analysis = result["data"]
    chart = result["chart"]

    rows = []

    for item in chart["data"]:
        if not isinstance(item, dict):
            continue

        category = item.get("category")
        raw_value = item.get("value")

        if category is None or raw_value is None:
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        
        raw_count = item.get("count")

        try:
            count = int(raw_count) if raw_count is not None else None
        except (TypeError, ValueError, OverflowError):
            count = None

        rows.append({
            "category": str(category),
            "value": value,
            "count": count,
        })


    if not rows:
        raise ValueError("Tidak ada nilai numerik yang dapat dilaporkan.")

    rows.sort(key=lambda item: item["value"], reverse=True)

    metric_label = str(
        analysis.get("metric_label")
        or chart.get("y_label")
        or "Metrik pertumbuhan"
    )

    metric = str(analysis.get("metric") or "")
    period = analysis.get("period")
    group_label = str(
        analysis.get("group_label")
        or chart.get("x_label")
        or "Kategori"
    )

    filters = analysis.get("filters")
    if not isinstance(filters, dict):
        filters = {}

    answer = str(result.get("answer") or "Ringkasan analisis tidak tersedia.")

    return {
        "question": str(question or "").strip(),
        "answer": answer,
        "metric": metric,
        "metric_label": metric_label,
        "group_label": group_label,
        "period": str(period) if period else "Tidak ditentukan",
        "filters": filters,
        "rows": rows,
        "chart_title": str(
            chart.get("title") or REPORT_SUBTITLE
        ),
        "generated_at": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "report_type": result.get("report_type", "comparison"),
        "statistics": analysis.get("statistics", {}),
        "row_count": analysis.get("row_count", 0),
        "selected_stat": analysis.get("selected_stat", "mean"),
        "stat_label": analysis.get("stat_label", "Rata-rata"),
    }


def _format_number(value: float) -> str:
    """Format angka untuk laporan berbahasa Indonesia."""

    return f"{value:,.4f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _filter_lines(filters: dict) -> list[str]:
    """Ambil filter yang benar-benar tersedia pada hasil analisis."""

    lines = []

    regional = filters.get("regional")
    species = filters.get("species")
    compartments = filters.get("compartments")
    clone = filters.get("clone")

    if regional:
        lines.append(f"Regional: {regional}")

    if species:
        lines.append(f"Spesies: {species}")

    if compartments:
        if isinstance(compartments, (list, tuple, set)):
            lines.append("Kompartemen: " + ", ".join(map(str, compartments)))
        else:
            lines.append(f"Kompartemen: {compartments}")

    if clone:
        lines.append(f"Clone: {clone}")

    if not lines:
        lines.append("Tidak ada filter khusus yang tercatat.")

    return lines



def _make_chart_image(report_data: dict) -> BytesIO:
    """Buat grafik perbandingan profesional untuk laporan PDF."""

    rows = report_data["rows"]
    labels = [row["category"] for row in rows]
    values = [row["value"] for row in rows]

    height = max(4.5, min(10.0, 0.48 * len(labels) + 2.0))

    fig, ax = plt.subplots(
        figsize=(10, height),
        layout="constrained",
    )

    green = "#2F6B45"
    dark_green = "#214D32"
    grid_color = "#DDE5DF"
    text_color = "#26352B"

    positions = list(range(len(labels)))

    bars = ax.barh(
        positions,
        values,
        height=0.65,
        color=green,
        edgecolor=dark_green,
        linewidth=0.4,
        zorder=3,
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=9, color=text_color)
    ax.invert_yaxis()

    ax.set_title(
        report_data["chart_title"],
        fontsize=13,
        fontweight="bold",
        color=dark_green,
        pad=16,
    )

    ax.set_xlabel(
        report_data["metric_label"],
        fontsize=9,
        color=text_color,
        labelpad=8,
    )

    ax.set_ylabel(
        report_data["group_label"],
        fontsize=9,
        color=text_color,
        labelpad=8,
    )

    max_value = max(values)
    min_value = min(values)

    # Beri ruang untuk label angka di ujung batang.
    if min_value >= 0:
        ax.set_xlim(left=0, right=max_value * 1.18 if max_value else 1)
    else:
        span = max_value - min_value
        ax.set_xlim(
            left=min_value - span * 0.05,
            right=max_value + span * 0.18,
        )

    value_labels = [_format_number(value) for value in values]

    ax.bar_label(
        bars,
        labels=value_labels,
        padding=5,
        fontsize=8,
        color=text_color,
    )

    ax.xaxis.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        color=grid_color,
        zorder=0,
    )

    ax.set_axisbelow(True)

    ax.tick_params(axis="x", labelsize=8, colors=text_color)
    ax.tick_params(axis="y", length=0, pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.spines["bottom"].set_color(grid_color)
    ax.spines["bottom"].set_linewidth(0.8)

    image_buffer = BytesIO()

    fig.savefig(
        image_buffer,
        format="png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    image_buffer.seek(0)
    return image_buffer



def _build_executive_conclusion(report: dict) -> list[str]:
    """Susun kesimpulan singkat dari hasil perbandingan aktual."""

    rows = report["rows"]

    if not rows:
        return ["Tidak tersedia data untuk menyusun kesimpulan."]

    winner = rows[0]
    conclusions = [
        (
            f"{winner['category']} menempati peringkat pertama "
            f"dengan nilai { _format_number(winner['value']) } "
            f"untuk metrik {report['metric_label']}."
        ),
        (
            f"Analisis mencakup {len(rows)} kelompok "
            f"pada periode {report['period']}."
        ),
    ]

    if len(rows) >= 2:
        runner_up = rows[1]
        difference = winner["value"] - runner_up["value"]

        conclusions.append(
            (
                f"Selisih nilai antara peringkat pertama "
                f"({winner['category']}) dan peringkat kedua "
                f"({runner_up['category']}) adalah "
                f"{_format_number(difference)}."
            )
        )

    conclusions.append(
        "Peringkat menunjukkan perbandingan nilai pada "
        "kelompok dan filter yang digunakan; hasil ini tidak "
        "dengan sendirinya membuktikan hubungan sebab-akibat."
    )

    return conclusions




# ============================================================
# PDF REPORT
# ============================================================



def generate_pdf_report(result: dict, question: str = "") -> BytesIO:
    """Laporan PDF dua halaman dengan ringkasan, peringkat, dan visualisasi."""

    from xml.sax.saxutils import escape
    from reportlab.platypus import KeepTogether

    report = _get_report_data(result, question)

    # Gunakan template khusus untuk laporan statistik satu periode.
    if report.get("report_type") == "single_period":
        return _generate_single_period_pdf(report)

    # Pertanyaan perbandingan tetap menggunakan template lama.
    rows = report.get("rows", [])

    if not rows:
        raise ValueError("Tidak ada data perbandingan untuk dilaporkan.")

    output = BytesIO()

    # Palet desain
    dark = "#12665D"
    green = "#178477"
    mint = "#DDF5EC"
    pale = "#F2F8F5"
    ink = "#263A36"
    muted = "#667873"
    border = "#D6E4DE"
    blue_pale = "#EEF7FC"
    blue = "#2584B8"

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.55 * cm,
        title=REPORT_TITLE,
        author="Tree Growth Analytics",
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportHero",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        textColor=colors.white,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="ReportHeroSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#D8F1E8"),
    ))
    styles.add(ParagraphStyle(
        name="SectionPro",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor(dark),
        spaceBefore=5,
        spaceAfter=5,
        keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        name="BodyPro",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.3,
        leading=11.2,
        textColor=colors.HexColor(ink),
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="SmallPro",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor(muted),
    ))
    styles.add(ParagraphStyle(
        name="KpiLabelPro",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor(muted),
    ))
    styles.add(ParagraphStyle(
        name="KpiValuePro",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor(dark),
    ))
    styles.add(ParagraphStyle(
        name="KpiNotePro",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor(muted),
    ))

    def safe(value):
        return escape(str(value if value is not None else "-"))

    def section(title):
        return Paragraph(safe(title.upper()), styles["SectionPro"])

    def styled_table(data, widths, font_size=7.5):
        table = Table(
            data,
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(dark)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(ink)),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(border)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.white,
                colors.HexColor(pale),
            ]),
        ]))
        return table

    winner = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None

    metric_label = report["metric_label"]
    group_label = report["group_label"]
    period = report["period"]
    unit_label = metric_label

    # --------------------------------------------------------
    # Footer dan nomor halaman
    # --------------------------------------------------------

    def draw_footer(canvas, doc):
        canvas.saveState()
        page_width, _ = A4

        canvas.setStrokeColor(colors.HexColor(border))
        canvas.setLineWidth(0.5)
        canvas.line(
            doc.leftMargin,
            1.1 * cm,
            page_width - doc.rightMargin,
            1.1 * cm,
        )

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(muted))
        canvas.drawString(
            doc.leftMargin,
            0.75 * cm,
            "TREE GROWTH ANALYTICS  |  LAPORAN ANALISIS",
        )
        canvas.drawRightString(
            page_width - doc.rightMargin,
            0.75 * cm,
            f"Halaman {doc.page}",
        )
        canvas.restoreState()

    story = []

    # ========================================================
    # HALAMAN 1 — RINGKASAN EKSEKUTIF
    # ========================================================

    header_left = [
        Paragraph("TREE GROWTH ANALYTICS", styles["ReportHero"]),
        Paragraph(
            "LAPORAN ANALISIS PERBANDINGAN METRIK",
            styles["ReportHeroSub"],
        ),
    ]

    header_right = [
        Paragraph(
            f"<b>Periode:</b> {safe(period)}"
            f"<br/><b>Metrik:</b> {safe(metric_label)}"
            f"<br/><b>Tanggal:</b> {safe(report['generated_at'])}",
            ParagraphStyle(
                "HeaderMeta",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=7,
                leading=10,
                textColor=colors.white,
                alignment=2,
            ),
        )
    ]

    header = Table(
        [[header_left, header_right]],
        colWidths=[10.1 * cm, 6.2 * cm],
    )
    header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(dark)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ROUNDEDCORNERS", [7, 7, 7, 7]),
    ]))
    story.append(header)
    story.append(Spacer(1, 0.28 * cm))

    # Kartu KPI
    kpi_rows = [[
        Paragraph("KELOMPOK TERATAS", styles["KpiLabelPro"]),
        Paragraph("NILAI TERTINGGI", styles["KpiLabelPro"]),
        Paragraph("KELOMPOK DIANALISIS", styles["KpiLabelPro"]),
    ], [
        Paragraph(safe(winner["category"]), styles["KpiValuePro"]),
        Paragraph(safe(_format_number(winner["value"])), styles["KpiValuePro"]),
        Paragraph(f"{len(rows)} kelompok", styles["KpiValuePro"]),
    ], [
        Paragraph("Peringkat #1", styles["KpiNotePro"]),
        Paragraph(safe(metric_label), styles["KpiNotePro"]),
        Paragraph(
            f"{safe(group_label)} · {safe(period)}",
            styles["KpiNotePro"],
        ),
    ]]

    kpi = Table(
        kpi_rows,
        colWidths=[5.43 * cm, 5.43 * cm, 5.44 * cm],
    )
    kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FBF5")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(border)),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor(border)),
        ("LINEBEFORE", (0, 0), (0, -1), 2, colors.HexColor(green)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi)
    story.append(Spacer(1, 0.23 * cm))

    # Panel pertanyaan
    question_text = safe(
        report.get("question") or question or "Pertanyaan tidak tersedia."
    )
    question_panel = Table([[
        [
            Paragraph(
                "PERTANYAAN ANALISIS",
                ParagraphStyle(
                    "QuestionLabel",
                    parent=styles["Normal"],
                    fontName="Helvetica-Bold",
                    fontSize=7,
                    leading=9,
                    textColor=colors.HexColor(blue),
                    spaceAfter=3,
                ),
            ),
            Paragraph(question_text, styles["BodyPro"]),
        ]
    ]], colWidths=[16.3 * cm])
    question_panel.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(blue_pale)),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D2E7F1")),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, colors.HexColor(blue)),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(question_panel)
    story.append(Spacer(1, 0.18 * cm))

    story.append(section("Ringkasan Eksekutif"))
    story.append(Paragraph(
        f"Laporan ini membandingkan <b>{len(rows)} kelompok</b> "
        f"berdasarkan metrik <b>{safe(metric_label)}</b> pada "
        f"periode <b>{safe(period)}</b>. Hasil berikut mengikuti "
        "nilai yang dikembalikan oleh mesin analitik backend.",
        styles["BodyPro"],
    ))

    # Grafik utama
    story.append(section("Visualisasi Perbandingan"))

    chart_buffer = _make_chart_image(report)
    main_chart = Image(
        chart_buffer,
        width=16.3 * cm,
        height=9.0 * cm,
        kind="proportional",
    )
    main_chart.hAlign = "CENTER"
    story.append(main_chart)
    story.append(Spacer(1, 0.1 * cm))

    story.append(section("Temuan Utama"))

    
    findings = [
        (
            f"{winner['category']} menempati peringkat pertama "
            f"dengan nilai {_format_number(winner['value'])} "
            f"untuk metrik {metric_label} pada periode {period}."
        )
    ]

    if runner_up is not None:
        gap = float(winner["value"]) - float(runner_up["value"])

        findings.append(
            f"{runner_up['category']} berada di peringkat kedua "
            f"dengan nilai {_format_number(runner_up['value'])}. "
            f"Selisih nilai terhadap kelompok teratas adalah "
            f"{_format_number(gap)}."
        )

        if float(runner_up["value"]) != 0:
            relative_gap = (
                gap / abs(float(runner_up["value"]))
            ) * 100

            findings.append(
                f"Nilai kelompok teratas lebih tinggi sekitar "
                f"{relative_gap:.1f}% dibandingkan peringkat kedua. "
                f"Persentase ini merupakan perbandingan relatif "
                f"antara dua nilai, bukan bukti hubungan sebab-akibat."
            )

    if len(rows) >= 2:
        lowest = rows[-1]
        spread = float(winner["value"]) - float(lowest["value"])

        findings.append(
            f"Rentang nilai dari kelompok teratas hingga "
            f"kelompok terbawah ({lowest['category']}) adalah "
            f"{_format_number(spread)}."
        )


    
    for index, finding in enumerate(findings, start=1):
        story.append(
            Paragraph(
                f"<b>{index}.</b> {safe(finding)}",
                styles["BodyPro"],
            )
        )



    # ========================================================
    # HALAMAN 2 — PERINGKAT, PROPORSI, METODOLOGI
    # ========================================================

    story.append(PageBreak())
    story.append(section("Tabel Peringkat"))

    ranking_data = [[
        "PERINGKAT",
        safe(group_label).upper(),
        safe(metric_label).upper(),
        "SELISIH DARI TERATAS",
    ]]

    for rank, row in enumerate(rows, start=1):
        difference = float(winner["value"]) - float(row["value"])
        label = str(row["category"])
        if rank == 1:
            label += " (Kelompok Teratas)"

        
    ranking_data = [[
        "PERINGKAT",
        safe(group_label).upper(),
        safe(metric_label).upper(),
        "JUMLAH OBSERVASI",
        "SELISIH DARI TERATAS",
    ]]

    for rank, row in enumerate(rows, start=1):
        difference = float(winner["value"]) - float(row["value"])
        label = str(row["category"])

        if rank == 1:
            label += " (Kelompok Teratas)"

        count_value = row.get("count")

        if count_value is not None:
            count_label = f"{int(count_value):,}".replace(",", ".")
        else:
            count_label = "-"

        ranking_data.append([
            str(rank),
            safe(label),
            safe(_format_number(row["value"])),
            safe(count_label),
            safe(_format_number(difference)),
        ])

        ranking = styled_table(
            ranking_data,
            [1.2 * cm, 4.0 * cm, 2.8 * cm, 2.5 * cm, 5.8 * cm],
            font_size=6.5,
        )
    ranking.setStyle(TableStyle([
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor(mint)),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("ALIGN", (0, 1), (0, -1), "CENTER"),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
    ]))
    story.append(ranking)
    story.append(Spacer(1, 0.3 * cm))

    # Donut chart hanya untuk nilai nonnegatif dengan total positif.
    numeric_values = [float(row["value"]) for row in rows]
    total = sum(numeric_values)

    story.append(section("Proporsi Relatif Nilai Kelompok"))

    if all(value >= 0 for value in numeric_values) and total > 0:
        sorted_pairs = sorted(
            zip(rows, numeric_values),
            key=lambda pair: pair[1],
            reverse=True,
        )

        top_count = min(4, len(sorted_pairs))
        donut_values = [
            value for _, value in sorted_pairs[:top_count]
        ]
        donut_labels = [
            str(row["category"]) for row, _ in sorted_pairs[:top_count]
        ]

        remainder = sum(
            value for _, value in sorted_pairs[top_count:]
        )
        if remainder > 0:
            donut_values.append(remainder)
            donut_labels.append(
                f"Lainnya ({len(sorted_pairs) - top_count} kelompok)"
            )

        fig, ax = plt.subplots(figsize=(4.0, 2.7), layout="constrained")
        palette = ["#12665D", "#178477", "#32B8A7", "#85D6C5", "#A8B7C8"]

        wedges, _ = ax.pie(
            donut_values,
            startangle=90,
            counterclock=False,
            colors=palette[:len(donut_values)],
            wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 1.2},
        )

        ax.text(
            0,
            0,
            f"TOTAL\n{_format_number(total)}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=dark,
        )
        ax.set_aspect("equal")

        donut_buffer = BytesIO()
        fig.savefig(
            donut_buffer,
            format="png",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
        donut_buffer.seek(0)

        legend_rows = []
        for label, value in zip(donut_labels, donut_values):
            share = 100 * value / total
            legend_rows.append([
                safe(label),
                f"{share:.1f}%",
            ])

        legend_data = [["KELOMPOK", "PROPORSI"]] + legend_rows
        legend_table = styled_table(
            legend_data,
            [5.4 * cm, 2.0 * cm],
            font_size=7,
        )

        donut_image = Image(
            donut_buffer,
            width=6.7 * cm,
            height=4.4 * cm,
            kind="proportional",
        )

        visual_row = Table(
            [[donut_image, legend_table]],
            colWidths=[8.0 * cm, 8.3 * cm],
        )
        visual_row.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(border)),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(visual_row)
        story.append(Paragraph(
            "Catatan: proporsi menunjukkan bagian setiap nilai terhadap "
            "jumlah nilai seluruh kelompok. Ini bukan kontribusi kausal "
            "terhadap pertumbuhan.",
            styles["SmallPro"],
        ))
    else:
        story.append(Paragraph(
            "Grafik proporsi tidak ditampilkan karena nilai tidak "
            "memenuhi syarat untuk membentuk proporsi yang bermakna. "
            "Gunakan tabel peringkat untuk membandingkan nilai.",
            styles["BodyPro"],
        ))

    story.append(Spacer(1, 0.22 * cm))
    story.append(section("Ruang Lingkup dan Filter"))

    scope_data = [["INFORMASI", "NILAI"]]
    scope_data.extend([
        ["Metrik", safe(metric_label)],
        ["Periode", safe(period)],
        ["Pengelompokan", safe(group_label)],
        ["Jumlah kelompok", str(len(rows))],
    ])

    for line in _filter_lines(report.get("filters", {})):
        if ":" in line:
            key, value = line.split(":", 1)
            scope_data.append([safe(key.strip()), safe(value.strip())])

    story.append(styled_table(
        scope_data,
        [4.3 * cm, 12.0 * cm],
        font_size=7.2,
    ))

    story.append(Spacer(1, 0.22 * cm))
    story.append(section("Interpretasi, Metodologi, dan Batasan"))

    story.append(Paragraph(
        f"Kelompok dengan nilai tertinggi adalah <b>{safe(winner['category'])}</b> "
        f"({_format_number(winner['value'])}). Hasil ini merupakan "
        "perbandingan deskriptif pada kelompok dan filter yang dipilih.",
        styles["BodyPro"],
    ))
    story.append(Paragraph(
        "Angka pada laporan berasal dari hasil analisis backend yang "
        "disediakan. Peringkat tidak dengan sendirinya membuktikan "
        "hubungan sebab-akibat. Interpretasi juga perlu mempertimbangkan "
        "jumlah observasi, kondisi lapangan, serta ruang lingkup data "
        "yang tersedia.",
        styles["BodyPro"],
    ))
    story.append(Paragraph(
        f"Dibuat otomatis oleh {safe(REPORT_TITLE)} pada "
        f"{safe(report['generated_at'])}.",
        styles["SmallPro"],
    ))

    document.build(
        story,
        onFirstPage=draw_footer,
        onLaterPages=draw_footer,
    )

    output.seek(0)
    return output



def _generate_single_period_pdf(report: dict) -> BytesIO:
    """Membuat PDF analisis statistik untuk satu periode pertumbuhan."""
    from xml.sax.saxutils import escape

    output = BytesIO()

    dark = "#12665D"
    green = "#178477"
    pale = "#F2F8F5"
    ink = "#263A36"
    muted = "#667873"

    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.6 * cm,
        title="Laporan Analisis Pertumbuhan Pohon",
        author="Tree Growth Analytics",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="SinglePeriodTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=25,
        textColor=colors.HexColor(dark),
        alignment=TA_LEFT,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SinglePeriodSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor(dark),
        spaceBefore=10,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SinglePeriodBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor(ink),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SinglePeriodSmall",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(muted),
    ))

    metric_label = escape(str(report.get("metric_label", "Metrik pertumbuhan")))
    period = escape(str(report.get("period", "Tidak ditentukan")))
    stat_label = escape(str(report.get("stat_label", "Rata-rata")))
    answer = escape(str(report.get("answer", "Ringkasan tidak tersedia.")))
    question = escape(str(report.get("question") or "Tidak tersedia"))
    row_count = int(report.get("row_count") or 0)
    statistics = report.get("statistics") or {}

    def fmt(value):
        if value is None:
            return "Tidak tersedia"
        try:
            return _format_number(float(value))
        except (TypeError, ValueError):
            return "Tidak tersedia"

    selected_stat = report.get("selected_stat", "mean")
    selected_value = statistics.get(selected_stat)
    filters = report.get("filters") or {}

    story = []

    # Header laporan
    story.append(Paragraph(
        "LAPORAN ANALISIS<br/>PERTUMBUHAN POHON",
        styles["SinglePeriodTitle"],
    ))
    story.append(Paragraph(
        f"Metrik: <b>{metric_label}</b> &nbsp;|&nbsp; "
        f"Periode: <b>{period}</b>",
        styles["SinglePeriodBody"],
    ))
    story.append(Spacer(1, 0.15 * cm))

    # KPI utama
    kpi_data = [
        [
            Paragraph("<b>NILAI UTAMA</b>", styles["SinglePeriodSmall"]),
            Paragraph("<b>JUMLAH BARIS DATA</b>", styles["SinglePeriodSmall"]),
            Paragraph("<b>PERIODE</b>", styles["SinglePeriodSmall"]),
        ],
        [
            Paragraph(
                f"<font size='16' color='{dark}'><b>{fmt(selected_value)}</b></font>"
                f"<br/>{stat_label}",
                styles["SinglePeriodBody"],
            ),
            Paragraph(
                f"<font size='16' color='{dark}'><b>{row_count:,}</b></font>"
                "<br/>baris data",
                styles["SinglePeriodBody"],
            ),
            Paragraph(
                f"<font size='13' color='{dark}'><b>{period}</b></font>",
                styles["SinglePeriodBody"],
            ),
        ],
    ]

    kpi_table = Table(kpi_data, colWidths=[5.4 * cm, 5.4 * cm, 5.4 * cm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(pale)),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D6E4DE")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6E4DE")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(kpi_table)

    story.append(Paragraph("Pertanyaan Analisis", styles["SinglePeriodSection"]))
    story.append(Paragraph(question, styles["SinglePeriodBody"]))

    story.append(Paragraph("Ringkasan Hasil", styles["SinglePeriodSection"]))
    story.append(Paragraph(answer, styles["SinglePeriodBody"]))

    # Statistik aktual yang diterima dari backend
    story.append(Paragraph("Tabel Statistik Metrik", styles["SinglePeriodSection"]))

    stats_rows = [
        ["STATISTIK", "NILAI"],
        ["Nilai minimum", fmt(statistics.get("min"))],
        ["Rata-rata", fmt(statistics.get("mean"))],
        ["Nilai maksimum", fmt(statistics.get("max"))],
        ["Jumlah sampel metrik", str(statistics.get("count", "Tidak tersedia"))],
    ]

    stats_table = Table(stats_rows, colWidths=[8.1 * cm, 8.1 * cm], repeatRows=1)
    stats_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(dark)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor(pale)]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6E4DE")),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(stats_table)

    # Grafik ringkasan minimum, rata-rata, dan maksimum
    chart_labels = ["Minimum", "Rata-rata", "Maksimum"]
    chart_keys = ["min", "mean", "max"]
    chart_values = []

    for key in chart_keys:
        try:
            chart_values.append(float(statistics[key]))
        except (KeyError, TypeError, ValueError):
            chart_values.append(None)

    valid_chart = [
        (label, value)
        for label, value in zip(chart_labels, chart_values)
        if value is not None
    ]

    if valid_chart:
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        labels = [item[0] for item in valid_chart]
        values = [item[1] for item in valid_chart]

        ax.bar(labels, values)
        ax.set_title(f"Ringkasan Statistik — {report.get('metric_label', 'Metrik')}")
        ax.set_ylabel(str(report.get("metric_label", "Nilai")))
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        chart_buffer = BytesIO()
        fig.savefig(chart_buffer, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        chart_buffer.seek(0)

        chart_image = Image(chart_buffer, width=16.0 * cm, height=6.8 * cm)
        chart_image.hAlign = "CENTER"

        story.append(Paragraph("Visualisasi Statistik", styles["SinglePeriodSection"]))
        story.append(chart_image)
        story.append(Paragraph(
            "Grafik memperlihatkan ringkasan minimum, rata-rata, dan maksimum "
            "dari statistik yang tersedia. Grafik ini bukan tren pertumbuhan "
            "antarperiode.",
            styles["SinglePeriodSmall"],
        ))

    # Ruang lingkup dan filter
    story.append(Paragraph("Ruang Lingkup dan Filter", styles["SinglePeriodSection"]))

    scope_rows = [
        ["INFORMASI", "NILAI"],
        ["Metrik", metric_label],
        ["Periode", period],
        ["Jumlah baris data", f"{row_count:,}"],
    ]

    for line in _filter_lines(filters):
        if ":" in line:
            key, value = line.split(":", 1)
            scope_rows.append([escape(key.strip()), escape(value.strip())])

    scope_table = Table(scope_rows, colWidths=[5.0 * cm, 11.2 * cm], repeatRows=1)
    scope_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(dark)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D6E4DE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor(pale)]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(scope_table)

    story.append(Paragraph("Interpretasi dan Batasan", styles["SinglePeriodSection"]))
    story.append(Paragraph(
        "Laporan ini menyajikan statistik deskriptif dari hasil analisis backend "
        "pada periode dan filter yang dipilih. Nilai minimum, rata-rata, dan "
        "maksimum menggambarkan sebaran data yang tersedia; nilai-nilai tersebut "
        "tidak dengan sendirinya menunjukkan perubahan pertumbuhan dari waktu "
        "ke waktu atau hubungan sebab-akibat. Jumlah baris data dan jumlah "
        "sampel metrik dapat berbeda jika ada nilai yang tidak valid.",
        styles["SinglePeriodBody"],
    ))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"Dibuat otomatis oleh {escape(REPORT_TITLE)} pada "
        f"{escape(str(report.get('generated_at', '')))}.",
        styles["SinglePeriodSmall"],
    ))

    def draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D6E4DE"))
        canvas.line(1.65 * cm, 1.2 * cm, A4[0] - 1.65 * cm, 1.2 * cm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(muted))
        canvas.drawString(1.65 * cm, 0.8 * cm, "TREE GROWTH ANALYTICS")
        canvas.drawRightString(A4[0] - 1.65 * cm, 0.8 * cm, f"Halaman {doc.page}")
        canvas.restoreState()

    document.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    output.seek(0)
    return output


def _generate_single_period_pptx(report: dict) -> BytesIO:
    """Membuat PowerPoint untuk analisis statistik satu periode."""
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.chart.data import CategoryChartData

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    GREEN = RGBColor(23, 69, 46)
    LEAF = RGBColor(47, 118, 80)
    PALE = RGBColor(244, 248, 244)
    WHITE = RGBColor(255, 255, 255)
    INK = RGBColor(35, 49, 40)
    MUTED = RGBColor(96, 112, 101)

    metric = str(report.get("metric_label") or "Metrik")
    period = str(report.get("period") or "Tidak ditentukan")
    question = str(report.get("question") or "Analisis statistik pertumbuhan pohon")
    answer = str(report.get("answer") or "Ringkasan analisis tidak tersedia.")
    statistics = report.get("statistics") or {}
    filters = report.get("filters") or {}
    selected_stat = str(report.get("selected_stat") or "mean")
    selected_value = statistics.get(selected_stat)
    row_count = report.get("row_count", 0)

    def fmt(value):
        if value is None:
            return "?"
        try:
            return f"{float(value):,.4f}"
        except (TypeError, ValueError):
            return str(value)

    def add_text(slide, text, x, y, w, h, size=16,
                 color=INK, bold=False, align=PP_ALIGN.LEFT):
        shape = slide.shapes.add_textbox(
            Inches(x), Inches(y), Inches(w), Inches(h)
        )
        tf = shape.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.text = str(text)
        p.alignment = align
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        return shape

    def add_card(slide, x, y, w, h, label, value):
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(x), Inches(y), Inches(w), Inches(h)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = PALE
        shape.line.color.rgb = RGBColor(210, 223, 213)
        add_text(slide, label, x + 0.15, y + 0.12, w - 0.3, 0.35,
                 12, MUTED, True)
        add_text(slide, value, x + 0.15, y + 0.52, w - 0.3, 0.55,
                 21, GREEN, True)

    # Slide 1: Ringkasan
    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE

    add_text(slide, REPORT_TITLE, 0.6, 0.35, 12.0, 0.45,
             14, LEAF, True)
    add_text(slide, "Laporan Analisis Statistik Pertumbuhan Pohon",
             0.6, 0.9, 12.0, 0.7, 27, GREEN, True)
    add_text(slide, f"Metrik: {metric} | Periode: {period}",
             0.6, 1.65, 12.0, 0.45, 15, MUTED)

    add_card(slide, 0.6, 2.35, 3.9, 1.35,
             f"Nilai {selected_stat}", fmt(selected_value))
    add_card(slide, 4.7, 2.35, 3.9, 1.35,
             "Jumlah baris data", f"{row_count:,}")
    add_card(slide, 8.8, 2.35, 3.9, 1.35,
             "Periode analisis", period)

    add_text(slide, "PERTANYAAN ANALISIS",
             0.6, 4.1, 12.0, 0.35, 13, LEAF, True)
    add_text(slide, question, 0.6, 4.45, 12.0, 0.75, 16)
    add_text(slide, "RINGKASAN HASIL",
             0.6, 5.35, 12.0, 0.35, 13, LEAF, True)
    add_text(slide, answer, 0.6, 5.7, 12.0, 1.0, 15)

    # Slide 2: Statistik dan grafik
    slide = prs.slides.add_slide(blank)
    add_text(slide, "RINGKASAN STATISTIK",
             0.6, 0.35, 12.0, 0.6, 25, GREEN, True)

    stat_labels = ["Minimum", "Rata-rata", "Maksimum", "Jumlah sampel"]
    stat_values = [
        fmt(statistics.get("min")),
        fmt(statistics.get("mean")),
        fmt(statistics.get("max")),
        str(statistics.get("count", "Tidak tersedia")),
    ]

    for i, (label, value) in enumerate(zip(stat_labels, stat_values)):
        y = 1.2 + i * 0.75
        add_text(slide, label, 0.8, y, 3.2, 0.45, 15, MUTED, True)
        add_text(slide, value, 4.0, y, 3.2, 0.45, 16, INK)

    chart_values = []
    chart_labels = []
    for label, key in [
        ("Minimum", "min"),
        ("Rata-rata", "mean"),
        ("Maksimum", "max"),
    ]:
        try:
            value = float(statistics[key])
            chart_labels.append(label)
            chart_values.append(value)
        except (KeyError, TypeError, ValueError):
            continue

    if chart_values:
        chart_data = CategoryChartData()
        chart_data.categories = chart_labels
        chart_data.add_series(metric, chart_values)
        chart = slide.shapes.add_chart(
            XL_CHART_TYPE.COLUMN_CLUSTERED,
            Inches(7.0), Inches(1.2), Inches(5.5), Inches(4.8),
            chart_data,
        ).chart
        chart.has_legend = False
        chart.has_title = True
        chart.chart_title.text_frame.text = f"Statistik {metric}"

    # Slide 3: Ruang lingkup dan filter
    slide = prs.slides.add_slide(blank)
    add_text(slide, "RUANG LINGKUP DAN FILTER",
             0.6, 0.35, 12.0, 0.6, 25, GREEN, True)

    scope_items = [
        ("Metrik", metric),
        ("Periode", period),
        ("Jumlah baris data", f"{row_count:,}"),
    ]

    for key, value in filters.items():
        scope_items.append((str(key), str(value)))

    for i, (label, value) in enumerate(scope_items):
        y = 1.2 + i * 0.55
        add_text(slide, label, 0.8, y, 3.5, 0.4, 14, MUTED, True)
        add_text(slide, value, 4.3, y, 8.0, 0.4, 14, INK)

    add_text(slide, "Catatan: laporan ini menggambarkan statistik pada "
             "periode yang dipilih, bukan tren antarperiode.",
             0.8, 6.2, 11.8, 0.55, 12, MUTED)

    output = BytesIO()
    prs.save(output)
    output.seek(0)
    return output


# ============================================================
# POWERPOINT REPORT
# ============================================================


def generate_pptx_report(result: dict, question: str = "") -> BytesIO:
    """Membuat presentasi analisis yang rapi, konsisten, dan dapat diedit."""
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    from pptx.enum.chart import XL_LABEL_POSITION
    from pptx.chart.data import CategoryChartData


    report = _get_report_data(result, question)

    if report.get("report_type") == "single_period":
        return _generate_single_period_pptx(report)

    rows = report.get("rows", [])

    
    if not rows:
        raise ValueError("Tidak ada data perbandingan untuk dibuatkan PowerPoint.")

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Palet konsisten: hijau sebagai identitas, warna terang untuk keterbacaan.
    GREEN = RGBColor(23, 69, 46)
    LEAF = RGBColor(47, 118, 80)
    MINT = RGBColor(221, 235, 224)
    PALE = RGBColor(244, 248, 244)
    WHITE = RGBColor(255, 255, 255)
    INK = RGBColor(35, 49, 40)
    MUTED = RGBColor(96, 112, 101)
    BORDER = RGBColor(210, 223, 213)
    GOLD = RGBColor(220, 174, 68)
    LIGHT_TEXT = RGBColor(229, 241, 231)

    metric = str(report.get("metric_label") or "Metrik analisis")
    period = str(report.get("period") or "Tidak ditentukan")
    group_label = str(report.get("group_label") or "Kelompok")
    chart_title = str(report.get("chart_title") or "Perbandingan hasil analisis")
    filters = report.get("filters") or {}
    question_text = str(question or report.get("question") or "Analisis perbandingan pertumbuhan pohon")
    conclusions = _build_executive_conclusion(report)
    winner = rows[0]
    runner_up = rows[1] if len(rows) > 1 else None
    winner_name = str(winner.get("category", "—"))
    group_count = len(rows)
    sample_count = sum(
        int(r["count"]) for r in rows
        if isinstance(r.get("count"), (int, float)) and r["count"] >= 0
    )
    margin = float(winner["value"]) - float(runner_up["value"]) if runner_up else None

    def fmt(value):
        if value is None:
            return "—"
        try:
            return f"{float(value):,.4f}"
        except (TypeError, ValueError):
            return str(value)

    def add_text(slide, text, x, y, w, h, size=14, color=INK,
                 bold=False, align=PP_ALIGN.LEFT, font="Aptos",
                 valign=MSO_ANCHOR.MIDDLE):
        shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = shape.text_frame
        tf.clear()
        tf.word_wrap = True
        tf.vertical_anchor = valign
        tf.margin_left = Inches(0.06)
        tf.margin_right = Inches(0.06)
        tf.margin_top = Inches(0.025)
        tf.margin_bottom = Inches(0.025)
        p = tf.paragraphs[0]
        p.text = str(text)
        p.alignment = align
        p.font.name = font
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        return shape

    def add_rect(slide, x, y, w, h, fill, line=None, radius=False):
        shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
        shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
        shape.line.color.rgb = line or fill
        if radius:
            try:
                shape.adjustments[0] = 0.08
            except (AttributeError, IndexError):
                pass
        return shape

    def add_base(slide, title, subtitle, page):
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = PALE
        add_rect(slide, 0, 0, 13.333, 1.38, GREEN)
        add_rect(slide, 0, 1.34, 13.333, 0.06, LEAF)
        add_text(slide, title, 0.62, 0.22, 12.0, 0.52, 23, WHITE, True)
        if subtitle:
            add_text(slide, subtitle, 0.66, 0.79, 11.9, 0.34, 10, LIGHT_TEXT)
        add_rect(slide, 0.62, 7.13, 12.08, 0.015, BORDER)
        add_text(slide, REPORT_TITLE, 0.64, 7.17, 8.0, 0.18, 8, GREEN, True)
        add_text(slide, f"{page:02d}", 12.0, 7.16, 0.62, 0.2, 8, GREEN, True, PP_ALIGN.RIGHT)

    def add_numbered_list(slide, items, x, y, w, line_h=0.58, size=12):
        for i, item in enumerate(items):
            yy = y + i * line_h
            add_rect(slide, x, yy + 0.035, 0.34, 0.34, LEAF, LEAF, radius=True)
            add_text(slide, str(i + 1), x, yy + 0.035, 0.34, 0.34, 10, WHITE, True, PP_ALIGN.CENTER)
            add_text(slide, item, x + 0.48, yy, w - 0.48, line_h - 0.02, size, INK)

    def add_card(slide, x, y, w, h, label, value, detail=""):
        add_rect(slide, x, y, w, h, WHITE, BORDER, radius=True)
        add_rect(slide, x, y, 0.075, h, LEAF)
        add_text(slide, label.upper(), x + 0.19, y + 0.12, w - 0.32, 0.24, 8, MUTED, True)
        add_text(slide, value, x + 0.19, y + 0.40, w - 0.32, 0.52, 19, GREEN, True)
        if detail:
            add_text(slide, detail, x + 0.19, y + h - 0.35, w - 0.32, 0.22, 8, MUTED)

    # 1 — Sampul: hijau dominan dengan panel informasi yang ringkas.
    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = GREEN
    add_rect(slide, 0, 0, 0.18, 7.5, LEAF)
    add_rect(slide, 0.78, 0.78, 0.09, 0.48, GOLD)
    add_text(slide, REPORT_TITLE, 1.08, 0.78, 10.8, 0.35, 14, LIGHT_TEXT, True)
    add_text(slide, "Laporan Analisis\nPerbandingan Pertumbuhan Pohon",
             1.02, 1.48, 11.0, 1.35, 30, WHITE, True, valign=MSO_ANCHOR.TOP)
    add_rect(slide, 1.04, 3.15, 11.15, 2.0, RGBColor(34, 86, 57), RGBColor(58, 111, 77), radius=True)
    add_text(slide, "PERTANYAAN ANALISIS", 1.35, 3.38, 10.4, 0.25, 9, LIGHT_TEXT, True)
    add_text(slide, question_text, 1.35, 3.75, 10.25, 0.82, 17, WHITE, True, valign=MSO_ANCHOR.TOP)
    add_text(slide, f"Metrik  ·  {metric}", 1.08, 5.62, 5.5, 0.32, 12, WHITE, True)
    add_text(slide, f"Periode  ·  {period}", 1.08, 6.02, 5.5, 0.32, 12, LIGHT_TEXT)
    add_text(slide, f"{group_count} kelompok dibandingkan", 7.3, 5.62, 4.7, 0.32, 12, WHITE, True, PP_ALIGN.RIGHT)
    add_text(slide, f"Dibuat {datetime.now().strftime('%d %B %Y · %H:%M')}",
             7.3, 6.02, 4.7, 0.32, 10, LIGHT_TEXT, align=PP_ALIGN.RIGHT)
    add_rect(slide, 1.08, 6.65, 11.0, 0.025, LEAF)

    # 2 — Ringkasan eksekutif.
    slide = prs.slides.add_slide(blank)
    add_base(slide, "Ringkasan Eksekutif", "Ikhtisar hasil berdasarkan data dan filter analisis.", 2)
    card_y, card_h, gap = 1.72, 1.38, 0.18
    card_w = (12.05 - gap * 3) / 4
    card_items = [
        ("Kelompok dianalisis", str(group_count), group_label),
        ("Nilai tertinggi", fmt(winner.get("value")), winner_name),
        ("Jumlah sampel", f"{sample_count:,}" if sample_count else "—", "Total observasi yang tersedia"),
        ("Selisih dua teratas", fmt(margin), metric),
    ]
    for i, item in enumerate(card_items):
        add_card(slide, 0.64 + i * (card_w + gap), card_y, card_w, card_h, *item)

    add_text(slide, "Pertanyaan analisis", 0.72, 3.42, 4.0, 0.3, 12, GREEN, True)
    add_rect(slide, 0.68, 3.78, 11.95, 0.86, MINT, MINT, radius=True)
    add_text(slide, question_text, 0.92, 3.92, 11.45, 0.58, 13, INK, True)
    add_text(slide, "Kesimpulan utama", 0.72, 4.93, 4.0, 0.3, 12, GREEN, True)
    add_numbered_list(slide, conclusions[:2], 0.75, 5.28, 11.7, 0.62, 11)

    # 3 — Ruang lingkup dan filter sebagai baris bernomor.
    slide = prs.slides.add_slide(blank)
    add_base(slide, "Ruang Lingkup dan Filter", "Konteks data yang digunakan dalam perbandingan.", 3)
    scope_items = [
        f"Metrik analisis: {metric}",
        f"Periode analisis: {period}",
        f"Kelompok perbandingan: {group_label}",
        f"Jumlah kelompok: {group_count}",
    ] + _filter_lines(filters)
    add_rect(slide, 0.68, 1.68, 11.95, 4.95, WHITE, BORDER, radius=True)
    add_numbered_list(slide, scope_items, 1.02, 1.98, 11.2, 0.56, 12)

    # 4 — Grafik dengan nilai yang mudah dibaca dan panel angka penjelas.
    slide = prs.slides.add_slide(blank)
    add_base(slide, "Visualisasi Perbandingan", chart_title, 4)

    chart_data = CategoryChartData()
    chart_data.categories = [str(r.get("category", "—")) for r in rows]
    chart_data.add_series(metric, [float(r["value"]) for r in rows])

    # Grafik utama dibuat selebar halaman seperti grafik pada laporan PDF.
    # Penjelasan ranking tidak ditempatkan sebagai panel samping yang berdesakan.
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.BAR_CLUSTERED,
        Inches(0.78), Inches(1.62), Inches(11.78), Inches(4.35),
        chart_data,
    ).chart
    chart.has_legend = False
    chart.has_title = False
    chart.chart_style = 10
    chart.category_axis.reverse_order = True
    chart.value_axis.minimum_scale = 0
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.tick_labels.font.name = "Aptos"
    chart.value_axis.tick_labels.font.size = Pt(9)
    chart.value_axis.tick_labels.font.color.rgb = MUTED
    chart.category_axis.tick_labels.font.name = "Aptos"
    chart.category_axis.tick_labels.font.size = Pt(10)
    chart.category_axis.tick_labels.font.color.rgb = INK
    chart.value_axis.major_gridlines.format.line.color.rgb = BORDER

    # Judul sumbu membantu pembaca memahami kategori dan satuan metrik.
    chart.value_axis.has_title = True
    chart.value_axis.axis_title.text_frame.text = metric
    chart.value_axis.axis_title.text_frame.paragraphs[0].font.name = "Aptos"
    chart.value_axis.axis_title.text_frame.paragraphs[0].font.size = Pt(9)
    chart.value_axis.axis_title.text_frame.paragraphs[0].font.color.rgb = MUTED
    chart.category_axis.has_title = True
    chart.category_axis.axis_title.text_frame.text = group_label
    chart.category_axis.axis_title.text_frame.paragraphs[0].font.name = "Aptos"
    chart.category_axis.axis_title.text_frame.paragraphs[0].font.size = Pt(9)
    chart.category_axis.axis_title.text_frame.paragraphs[0].font.color.rgb = MUTED

    series = chart.series[0]
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = LEAF
    series.format.line.fill.background()
    series.has_data_labels = True
    # Paksa nilai numerik benar-benar tampil pada ujung setiap batang.
    # has_data_labels=True saja tidak selalu mengaktifkan show_value.
    series.data_labels.show_value = True
    series.data_labels.show_category_name = False
    series.data_labels.show_series_name = False
    series.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
    series.data_labels.number_format = "0.0000"
    series.data_labels.font.name = "Aptos"
    series.data_labels.font.size = Pt(9)
    series.data_labels.font.bold = True
    series.data_labels.font.color.rgb = INK

    # Ringkasan temuan di bawah grafik mengikuti gaya laporan PDF.
    add_text(slide, "TEMUAN UTAMA", 0.82, 6.08, 2.2, 0.24, 10, GREEN, True)
    top_value = fmt(winner.get("value"))
    second = rows[1] if len(rows) > 1 else None
    if second is not None:
        gap_value = float(winner["value"]) - float(second["value"])
        finding_lines = [
            f"1. {winner_name} berada di peringkat pertama dengan nilai {top_value} {metric}.",
            f"2. {second.get('category', '—')} berada di peringkat kedua dengan nilai {fmt(second.get('value'))}; selisihnya {fmt(gap_value)}.",
            f"3. Rentang nilai seluruh kelompok adalah {fmt(max(float(r['value']) for r in rows) - min(float(r['value']) for r in rows))}.",
        ]
    else:
        finding_lines = [
            f"1. {winner_name} memiliki nilai tertinggi, yaitu {top_value} {metric}.",
        ]
    for i, line in enumerate(finding_lines):
        add_text(slide, line, 0.88, 6.36 + i * 0.23, 11.65, 0.21, 8, INK)

    # 5 dst. — Tabel peringkat dengan format angka seragam.
    page = 5
    chunk_size = 10
    for start_index in range(0, len(rows), chunk_size):
        chunk = rows[start_index:start_index + chunk_size]
        slide = prs.slides.add_slide(blank)
        add_base(slide, "Tabel Peringkat" if start_index == 0 else "Tabel Peringkat (Lanjutan)",
                 f"Metrik: {metric}  ·  Kelompok: {group_label}", page)
        headers = ["Peringkat", group_label, metric, "Jumlah sampel", "Selisih dari tertinggi"]
        table = slide.shapes.add_table(
            len(chunk) + 1, len(headers), Inches(0.68), Inches(1.72),
            Inches(11.95), Inches(min(4.8, 0.43 * (len(chunk) + 1)))
        ).table
        widths = [1.15, 3.05, 2.15, 1.8, 3.8]
        for col, width in enumerate(widths):
            table.columns[col].width = Inches(width)
        for col, heading in enumerate(headers):
            cell = table.cell(0, col)
            cell.text = heading
            cell.fill.solid()
            cell.fill.fore_color.rgb = GREEN
            cell.margin_left = Inches(0.07)
            cell.margin_right = Inches(0.07)
            for p in cell.text_frame.paragraphs:
                p.font.name = "Aptos"
                p.font.size = Pt(9)
                p.font.bold = True
                p.font.color.rgb = WHITE
                p.alignment = PP_ALIGN.LEFT if col == 1 else PP_ALIGN.CENTER
        for i, row in enumerate(chunk, start=1):
            diff = float(winner["value"]) - float(row["value"])
            count = row.get("count")
            vals = [
                str(start_index + i),
                str(row.get("category", "—")),
                fmt(row.get("value")),
                f"{int(count):,}" if isinstance(count, (int, float)) else "—",
                fmt(diff),
            ]
            for col, value in enumerate(vals):
                cell = table.cell(i, col)
                cell.text = value
                cell.fill.solid()
                cell.fill.fore_color.rgb = MINT if (start_index + i == 1) else (WHITE if i % 2 else PALE)
                cell.margin_left = Inches(0.08)
                cell.margin_right = Inches(0.08)
                cell.margin_top = Inches(0.055)
                cell.margin_bottom = Inches(0.055)
                for p in cell.text_frame.paragraphs:
                    p.font.name = "Aptos"
                    p.font.size = Pt(9)
                    p.font.bold = (start_index + i == 1)
                    p.font.color.rgb = GREEN if (start_index + i == 1) else INK
                    p.alignment = PP_ALIGN.LEFT if col == 1 else PP_ALIGN.CENTER
        numeric_values = [float(r["value"]) for r in rows]
        add_text(slide, f"Rentang nilai: {fmt(min(numeric_values))} – {fmt(max(numeric_values))}",
                 0.82, 6.55, 11.4, 0.3, 10, GREEN, True)
        page += 1

    # Slide terakhir — temuan bernomor, bukan bullet.
    slide = prs.slides.add_slide(blank)
    add_base(slide, "Interpretasi dan Kesimpulan",
             "Temuan bersifat deskriptif dan tidak membuktikan sebab-akibat.", page)
    add_rect(slide, 0.68, 1.68, 11.95, 4.25, WHITE, BORDER, radius=True)
    add_text(slide, "Temuan analisis", 0.98, 1.88, 10.9, 0.32, 14, GREEN, True)
    add_numbered_list(slide, conclusions, 1.0, 2.38, 11.0, 0.68, 11)
    add_rect(slide, 0.68, 6.08, 11.95, 0.62, MINT, MINT, radius=True)
    add_text(slide,
             "Metodologi: angka mengikuti hasil analisis backend, filter, dan data yang tersedia.",
             0.92, 6.18, 11.45, 0.38, 9, GREEN, True)

    output = BytesIO()
    prs.save(output)
    output.seek(0)
    return output

# ============================================================
# MAI REPORT - TEMPLATE PDF KHUSUS
# ============================================================

def generate_mai_pdf_report(report: dict, question: str = "") -> BytesIO:
    """Membuat PDF khusus laporan MAI dari hasil endpoint MAI aktual."""
    from xml.sax.saxutils import escape

    if not isinstance(report, dict):
        raise ValueError("Data laporan MAI harus berupa dictionary.")

    details = report.get("details") or []
    summary = report.get("summary") or {}
    yearly = report.get("by_planting_year") or []
    methodology = report.get("methodology") or {}

    if not details:
        raise ValueError("Tidak ada detail data MAI untuk dibuat menjadi PDF.")

    output = BytesIO()

    dark = "#12665D"
    green = "#178477"
    pale = "#F2F8F5"
    mint = "#DDF5EC"
    ink = "#263A36"
    muted = "#667873"
    border = "#D6E4DE"

    document = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=1.2 * cm,
        leftMargin=1.2 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.2 * cm,
        title="Laporan Mean Annual Increment (MAI)",
        author="Tree Growth Analytics",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="MAIReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        alignment=TA_LEFT,
        textColor=colors.HexColor(dark),
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="MAIReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor(muted),
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="MAIReportSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor(dark),
        spaceBefore=9,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="MAIReportCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.5,
        leading=8,
        textColor=colors.HexColor(ink),
    ))
    styles.add(ParagraphStyle(
        name="MAIReportSmall",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor(muted),
    ))

    def fmt(value, digits=2):
        try:
            if value is None:
                return "-"
            return f"{float(value):,.{digits}f}"
        except (TypeError, ValueError):
            return "-"

    def cell(value):
        return Paragraph(escape(str(value if value is not None else "-")), styles["MAIReportCell"])

    story = [
        Paragraph("LAPORAN MEAN ANNUAL INCREMENT (MAI)", styles["MAIReportTitle"]),
        Paragraph(
            "TREE GROWTH ANALYTICS · Laporan berdasarkan data pengukuran aktual",
            styles["MAIReportSubtitle"],
        ),
        Paragraph(
            f"<b>Permintaan:</b> {escape(question or 'Laporan MAI')}",
            styles["MAIReportSmall"],
        ),
        Spacer(1, 6),
    ]

    threshold = summary.get("threshold_m3_ha_year", 25)
    summary_rows = [
        [cell("Kombinasi Kompartemen–Kohort"), cell("Luas Kohort (ha)"),
         cell("MAI Tertimbang (m³/ha/tahun)"), cell("MAI Minimum"),
         cell("MAI Maksimum"), cell("Status Baik"), cell("Di Bawah Ambang")],
        [
            cell(summary.get("cohort_compartment_count", 0)),
            cell(fmt(summary.get("cohort_area_ha"))),
            cell(fmt(summary.get("mai_weighted_m3_ha_year"), 3)),
            cell(fmt(summary.get("mai_min_m3_ha_year"), 3)),
            cell(fmt(summary.get("mai_max_m3_ha_year"), 3)),
            cell(summary.get("good_count", 0)),
            cell(summary.get("below_threshold_count", 0)),
        ],
    ]
    summary_table = Table(
        summary_rows,
        colWidths=[4.1*cm, 2.3*cm, 3.5*cm, 2.4*cm, 2.4*cm, 2.0*cm, 2.5*cm],
        repeatRows=1,
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(dark)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor(pale)),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(border)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([
        Paragraph("1. RINGKASAN MAI", styles["MAIReportSection"]),
        summary_table,
    ])

    if yearly:
        story.append(Paragraph("2. REKAPITULASI PER TAHUN TANAM", styles["MAIReportSection"]))
        yearly_rows = [[
            cell("Tahun Tanam"), cell("Jumlah Kompartemen"),
            cell("Luas Kohort (ha)"), cell("MAI Tertimbang"),
            cell("MAI Minimum"), cell("MAI Maksimum"),
            cell("Baik"), cell("Di Bawah Ambang"),
        ]]
        for item in yearly:
            yearly_rows.append([
                cell(item.get("planting_year", "-")),
                cell(item.get("compartment_count", 0)),
                cell(fmt(item.get("cohort_area_ha"))),
                cell(fmt(item.get("mai_weighted_m3_ha_year"), 3)),
                cell(fmt(item.get("mai_min_m3_ha_year"), 3)),
                cell(fmt(item.get("mai_max_m3_ha_year"), 3)),
                cell(item.get("good_count", 0)),
                cell(item.get("below_threshold_count", 0)),
            ])
        yearly_table = Table(yearly_rows, colWidths=[
            2.2*cm, 2.5*cm, 2.5*cm, 3.1*cm,
            2.5*cm, 2.5*cm, 1.5*cm, 2.5*cm,
        ], repeatRows=1)
        yearly_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(green)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                colors.white, colors.HexColor(pale)
            ]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(border)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(yearly_table)

    story.append(Paragraph("3. DETAIL MAI PER KOMPARTEMEN", styles["MAIReportSection"]))
    detail_rows = [[
        cell("No."), cell("Regional"), cell("Kompartemen"),
        cell("Tahun Tanam"), cell("Luas (ha)"), cell("Umur"),
        cell("Periode"), cell("Spesies"), cell("Klon"),
        cell("Pohon/ha"), cell("Volume/Pohon"),
        cell("Volume Estimasi/ha"), cell("MAI"),
        cell("Status"),
    ]]

    for index, item in enumerate(details, start=1):
        detail_rows.append([
            cell(index),
            cell(item.get("regional_name") or item.get("regional_id", "-")),
            cell(item.get("compartment_code") or item.get("compartment_id", "-")),
            cell(item.get("planting_year", "-")),
            cell(fmt(item.get("area_ha"))),
            cell(item.get("age_months", "-")),
            cell(item.get("growth_period", "-")),
            cell(item.get("species", "-")),
            cell(item.get("clone", "-")),
            cell(item.get("trees_per_ha", "-")),
            cell(fmt(item.get("mean_volume_m3_tree"), 4)),
            cell(fmt(item.get("estimated_volume_m3_ha"), 2)),
            cell(fmt(item.get("mai_m3_ha_year"), 3)),
            cell(item.get("status", "-")),
        ])

    detail_table = Table(detail_rows, colWidths=[
        0.7*cm, 2.0*cm, 1.8*cm, 1.5*cm, 1.2*cm, 1.0*cm,
        1.1*cm, 3.2*cm, 3.0*cm, 1.4*cm, 1.7*cm,
        2.0*cm, 1.7*cm, 1.7*cm,
    ], repeatRows=1)
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(dark)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor(pale)
        ]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor(border)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(detail_table)

    story.append(Paragraph("4. METODE DAN CATATAN", styles["MAIReportSection"]))
    story.append(Paragraph(
        "Ambang status Baik: MAI ≥ "
        + escape(fmt(threshold, 2))
        + " m³/ha/tahun. MAI dihitung dari estimasi volume per hektare "
        "dibagi umur pengamatan dalam tahun.",
        styles["MAIReportSmall"],
    ))
    for key in ("source_metric", "estimated_volume_formula", "mai_formula",
                "weighted_formula", "density_note", "area_note"):
        if methodology.get(key):
            story.append(Paragraph(
                f"<b>{escape(key.replace('_', ' ').capitalize())}:</b> "
                f"{escape(str(methodology[key]))}",
                styles["MAIReportSmall"],
            ))

    document.build(story)
    output.seek(0)
    return output
