import re

def resolve_context(question, history):
    q = question.strip()
    lower = q.lower()

    periods = r"(?<![a-z0-9])p(6|12|18|24|30|36|42|48|54|60|66)(?![a-z0-9])"
    explicit_period = re.search(periods, q, re.I)
    explicit_phi = re.search(r"(?<!\w)phi(?!\w)", q, re.I)

    growth_terms = (
        "pertumbuhan", "growth", "pohon", "dbh", "diameter",
        "tinggi", "height", "volume", "klon", "clone",
        "spesies", "species", "kompartemen", "compartment",
        "wilayah", "regional", "berdasarkan", "bandingkan",
        "perbandingan", "rata-rata", "rata rata"
    )
    followup_terms = (
        "ganti", "ubah", "sekarang", "lalu", "kalau",
        "bagaimana dengan", "berdasarkan", "bandingkan",
        "bandingkan berdasarkan", "yang sama", "periode sebelumnya"
    )
    unrelated_terms = (
        "apa itu pinns", "jelaskan pinns", "apa itu machine learning",
        "apa itu random forest"
    )

    if any(term in lower for term in unrelated_terms):
        return q

    is_followup = any(term in lower for term in followup_terms)
    has_growth_context = any(term in lower for term in growth_terms)

    if not is_followup or not has_growth_context:
        return q

    previous = next(
        (
            str(item.get("content", "")).strip()
            for item in reversed(history)
            if item.get("role") == "user"
            and str(item.get("content", "")).strip()
            and any(
                term in str(item.get("content", "")).lower()
                for term in ("pertumbuhan", "growth", "pohon", "volume",
                             "dbh", "diameter", "tinggi", "height",
                             "bandingkan", "perbandingan")
            )
        ),
        ""
    )

    if not previous:
        return q

    additions = []

    if not explicit_period and not explicit_phi:
        old_period = re.search(periods, previous, re.I)
        old_phi = re.search(r"(?<!\w)phi(?!\w)", previous, re.I)
        if old_period:
            additions.append(f"periode P{old_period.group(1)}")
        elif old_phi:
            additions.append("periode PHI")

    metric_patterns = (
        (r"\b(?:volume|volume_m3_tree)\b", "volume pohon"),
        (r"\b(?:dbh|diameter|dbh_cm)\b", "diameter pohon"),
        (r"\b(?:tinggi|height|height_m)\b", "tinggi pohon"),
        (r"\b(?:penyakit|disease_score)\b", "skor penyakit"),
    )
    current_has_metric = any(
        re.search(pattern, lower, re.I) for pattern, _ in metric_patterns
    )
    if not current_has_metric:
        for pattern, label in metric_patterns:
            if re.search(pattern, previous, re.I):
                additions.append(f"metrik {label}")
                break

    group_patterns = (
        (r"\b(?:klon|clone)\b", "klon"),
        (r"\b(?:spesies|species)\b", "spesies"),
        (r"\b(?:kompartemen|compartment)\b", "kompartemen"),
        (r"\b(?:wilayah|regional)\b", "wilayah"),
    )
    current_has_group = bool(re.search(
        r"\b(?:berdasarkan|group by|kelompok|pengelompokan)\b", lower
    ))
    if not current_has_group:
        for pattern, label in group_patterns:
            if re.search(pattern, previous, re.I):
                additions.append(f"berdasarkan {label}")
                break

    if additions:
        q += " " + " ".join(additions)

    return q


cases = [
    (
        "Bandingkan pertumbuhan volume pohon pada periode P18",
        [],
        "Bandingkan pertumbuhan volume pohon pada periode P18",
    ),
    (
        "Ganti pengelompokan berdasarkan klon",
        [{"role": "user", "content":
          "Bandingkan pertumbuhan volume pohon pada periode P18"}],
        "P18",
    ),
    (
        "Bandingkan pertumbuhan volume pohon pada periode P24",
        [{"role": "user", "content":
          "Bandingkan pertumbuhan volume pohon pada periode P18"}],
        "P24",
    ),
    (
        "Apa itu PINNs?",
        [{"role": "user", "content":
          "Bandingkan pertumbuhan volume pohon pada periode P18"}],
        "Apa itu PINNs?",
    ),
]

for question, history, expected in cases:
    result = resolve_context(question, history)
    passed = expected in result
    print(f"{'LULUS' if passed else 'GAGAL'} | {question}")
    print(f"  Hasil: {result}")

print("\n=== UJI KONTEKS LANJUTAN ===")

extra_cases = [
    (
        "Bandingkan pertumbuhan volume pohon pada periode P18",
        [{"role": "user", "content":
          "Bandingkan pertumbuhan volume pohon pada periode P18 berdasarkan spesies"}],
        ("P18", "volume pohon", "spesies"),
    ),
    (
        "Bandingkan pertumbuhan diameter pohon pada periode P24",
        [{"role": "user", "content":
          "Bandingkan pertumbuhan volume pohon pada periode P18 berdasarkan spesies"}],
        ("P24", "diameter pohon", "spesies"),
    ),
    (
        "Ganti pengelompokan berdasarkan klon",
        [{"role": "user", "content":
          "Bandingkan pertumbuhan volume pohon pada periode P18 berdasarkan spesies"}],
        ("P18", "volume pohon", "klon"),
    ),
    (
        "Apa itu PINNs?",
        [{"role": "user", "content":
          "Bandingkan pertumbuhan volume pohon pada periode P18 berdasarkan spesies"}],
        ("Apa itu PINNs?",),
    ),
]

for question, history, expected_parts in extra_cases:
    result = resolve_context(question, history)
    passed = all(part.lower() in result.lower() for part in expected_parts)
    print(f"{'LULUS' if passed else 'PERLU PERBAIKAN'} | {question}")
    print(f"  Hasil: {result}")
