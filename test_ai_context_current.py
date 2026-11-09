import re

def resolve_context(question, history):
    question = question.strip()

    explicit_period = re.search(
        r"(?<![a-z0-9])p\d+(?![a-z0-9])",
        question,
        re.IGNORECASE,
    )
    explicit_phi = re.search(
        r"(?<![a-z0-9])phi(?![a-z0-9])",
        question,
        re.IGNORECASE,
    )

    if not explicit_period and not explicit_phi:
        current_lower = question.lower().strip()

        followup_patterns = (
            r"^\s*(?:sekarang|ganti|ubah|lalu|bagaimana kalau|"
            r"kalau|bagaimana dengan)\b",
            r"\b(?:periode|period)\s+(?:sebelumnya|yang sama)\b",
        )
        is_followup = any(
            re.search(pattern, current_lower)
            for pattern in followup_patterns
        )

        if is_followup:
            for item in reversed(history):
                if item.get("role") != "user":
                    continue

                previous_question = str(
                    item.get("content", "")
                ).strip()

                period_match = re.search(
                    r"(?<![a-z0-9])p(6|12|18|24|30|36|42|48|54|60|66)(?![a-z0-9])",
                    previous_question,
                    re.IGNORECASE,
                )
                phi_match = re.search(
                    r"(?<!\w)phi(?!\w)",
                    previous_question,
                    re.IGNORECASE,
                )

                if period_match:
                    question = (
                        f"{question} pada periode "
                        f"P{period_match.group(1)}"
                    )
                    break

                if phi_match:
                    question = f"{question} pada periode PHI"
                    break

    return question


tests = [
    {
        "name": "Pertanyaan awal P18",
        "question": "Bandingkan volume pohon berdasarkan spesies pada P18",
        "history": [],
    },
    {
        "name": "Ganti pengelompokan",
        "question": "Ganti pengelompokan berdasarkan klon",
        "history": [
            {
                "role": "user",
                "content": "Bandingkan volume pohon berdasarkan spesies pada P18",
            }
        ],
    },
    {
        "name": "Periode baru P24",
        "question": "Bandingkan tinggi pohon pada P24",
        "history": [
            {
                "role": "user",
                "content": "Bandingkan volume pohon berdasarkan spesies pada P18",
            }
        ],
    },
    {
        "name": "Pertanyaan tidak terkait",
        "question": "Apa itu PINNs?",
        "history": [
            {
                "role": "user",
                "content": "Bandingkan volume pohon berdasarkan spesies pada P18",
            }
        ],
    },
]

for test in tests:
    print(f"\n[{test['name']}]")
    print("Input :", test["question"])
    print("Output:", resolve_context(test["question"], test["history"]))
