from pathlib import Path
from datetime import datetime
import re

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8-sig")

start_marker = "    # Resolusi konteks periode dari pertanyaan sebelumnya."
end_marker = "    q = question.lower()"

start = source.find(start_marker)
end = source.find(end_marker, start)

if start < 0 or end < 0:
    raise SystemExit("STOP: batas blok tidak ditemukan. File utama tidak diubah.")

end += len(end_marker)

replacement = r'''    # Resolusi konteks percakapan dengan prioritas pertanyaan terbaru.
    question = request.question.strip()
    current_lower = question.lower()

    period_pattern = (
        r"(?<![a-z0-9])p(6|12|18|24|30|36|42|48|54|60|66)"
        r"(?![a-z0-9])"
    )
    metric_patterns = (
        (r"\b(?:volume_m3_tree|volume per pohon|volume pohon|volume)\b",
         "volume pohon"),
        (r"\b(?:dbh_cm|dbh|diameter batang|diameter pohon|diameter)\b",
         "diameter pohon"),
        (r"\b(?:height_m|height|tinggi pohon|tinggi)\b",
         "tinggi pohon"),
        (r"\b(?:disease_score|disease score|skor penyakit)\b",
         "skor penyakit"),
        (r"\b(?:condition_score|condition score|skor kondisi)\b",
         "skor kondisi"),
        (r"\b(?:trees_per_ha|trees per ha|kerapatan pohon)\b",
         "kerapatan pohon"),
    )
    group_patterns = (
        (r"\bberdasarkan\s+(?:jenis\s+)?(?:klon|clone)\b", "klon"),
        (r"\bberdasarkan\s+(?:spesies|species)\b", "spesies"),
        (r"\bberdasarkan\s+(?:compartment|kompartemen|comp)\b",
         "kompartemen"),
        (r"\bberdasarkan\s+(?:wilayah|regional)\b", "wilayah"),
    )

    explicit_period = re.search(period_pattern, question, re.I)
    explicit_phi = re.search(r"(?<!\w)phi(?!\w)", question, re.I)
    current_metric = next(
        (label for pattern, label in metric_patterns
         if re.search(pattern, current_lower, re.I)),
        None,
    )
    current_group = next(
        (label for pattern, label in group_patterns
         if re.search(pattern, current_lower, re.I)),
        None,
    )

    unrelated_question = re.search(
        r"\b(?:apa itu|jelaskan|definisi)\s+"
        r"(?:pinns|random forest|machine learning)\b",
        current_lower,
    )
    followup = re.search(
        r"\b(?:ganti|ubah|sekarang|lalu|kalau|lanjut|"
        r"berdasarkan|pengelompokan|kelompokkan|yang sama)\b",
        current_lower,
    )

    if not unrelated_question and followup:
        previous_questions = [
            str(item.get("content", "")).strip()
            for item in reversed(request.history)
            if item.get("role") == "user"
            and str(item.get("content", "")).strip()
        ]

        previous_questions = [
            previous for previous in previous_questions
            if re.search(
                r"\b(?:bandingkan|perbandingan|pertumbuhan|growth|"
                r"volume|diameter|dbh|tinggi|height|spesies|species|"
                r"klon|clone|kompartemen|compartment|regional|wilayah)\b",
                previous,
                re.I,
            )
        ]

        additions = []

        if not explicit_period and not explicit_phi:
            for previous in previous_questions:
                old_period = re.search(period_pattern, previous, re.I)
                old_phi = re.search(r"(?<!\w)phi(?!\w)", previous, re.I)
                if old_period:
                    additions.append(f"periode P{old_period.group(1)}")
                    break
                if old_phi:
                    additions.append("periode PHI")
                    break

        if current_metric is None:
            for previous in previous_questions:
                old_metric = next(
                    (label for pattern, label in metric_patterns
                     if re.search(pattern, previous, re.I)),
                    None,
                )
                if old_metric:
                    additions.append(f"metrik {old_metric}")
                    break

        if current_group is None:
            for previous in previous_questions:
                old_group = next(
                    (label for pattern, label in group_patterns
                     if re.search(pattern, previous, re.I)),
                    None,
                )
                if old_group:
                    additions.append(f"berdasarkan {old_group}")
                    break

        if additions:
            if not re.search(
                r"\b(?:bandingkan|perbandingan|dibandingkan|dibanding)\b",
                current_lower,
            ):
                question = "Bandingkan " + question
            question += " " + " ".join(additions)

    q = question.lower()'''

updated = source[:start] + replacement + source[end:]

if "    # Resolusi konteks percakapan dengan prioritas pertanyaan terbaru." not in updated:
    raise SystemExit("STOP: marker pengganti tidak ditemukan. File utama tidak diubah.")

try:
    compile(updated, str(path), "exec")
except SyntaxError as exc:
    raise SystemExit(
        f"STOP: validasi sintaks gagal pada baris {exc.lineno}. "
        "File utama tidak diubah."
    )

backup = path.with_name(
    f"main_new.before_context_fix_{datetime.now():%Y%m%d_%H%M%S}.py"
)
backup.write_text(source, encoding="utf-8")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: blok resolver diganti.")
print(f"Backup: {backup}")
print("Validasi sintaks sebelum penulisan: LULUS")
