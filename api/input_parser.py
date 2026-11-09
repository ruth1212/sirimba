
# api/input_parser.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ============================================================
# 1. STRUKTUR INTENT
# ============================================================

@dataclass
class IntentSpec:
    name: str
    description: str
    examples: list[str]
    handler_key: str
    threshold: float = 0.48


@dataclass
class ParsedQuery:
    original_text: str
    normalized_text: str
    entities: dict[str, Any] = field(default_factory=dict)
    intent: Optional[str] = None
    similarity: float = 0.0
    second_similarity: float = 0.0
    status: str = "unmatched"
    handler_key: Optional[str] = None


# ============================================================
# 2. NORMALISASI TEKS
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# 3. PEMISAHAN MULTI-QUERY
# ============================================================

# Pemisah kuat: titik, titik koma, baris baru, dan pertanyaan baru
STRONG_BOUNDARY = re.compile(
    r"[;.!?\n]+"
)

# Konjungsi yang boleh menjadi kandidat pemisah.
# Pemisahan hanya dilakukan bila sesudahnya terdapat pola klausa
# baru, bukan pada setiap kemunculan kata "dan".
CLAUSE_START = re.compile(
    r"\b(?:"
    r"dan\s+(?=(?:bagaimana|apa|berapa|jelaskan|tampilkan|"
    r"cari|unduh|download|buat|hitung|bandingkan|tolong)\b)"
    r"|serta\s+(?=(?:bagaimana|apa|berapa|jelaskan|tampilkan|"
    r"cari|unduh|download|buat|hitung)\b)"
    r"|lalu\s+"
    r"|kemudian\s+"
    r"|selanjutnya\s+"
    r")",
    flags=re.IGNORECASE,
)


def split_queries(text: str) -> list[str]:
    """
    Memecah teks menjadi kandidat query.

    Tidak memisahkan semua kata 'dan', sebab 'bandingkan DBH dan
    tinggi' seharusnya tetap satu query.
    """
    if not text or not text.strip():
        return []

    # Tahap pertama: tanda baca kuat.
    chunks = [
        part.strip(" \t\r-")
        for part in STRONG_BOUNDARY.split(text)
        if part.strip(" \t\r-")
    ]

    # Tahap kedua: konjungsi dengan indikasi klausa baru.
    results: list[str] = []

    for chunk in chunks:
        parts = CLAUSE_START.split(chunk)

        for part in parts:
            cleaned = part.strip(" \t\r,.-")
            if cleaned:
                results.append(cleaned)

    # Hilangkan duplikasi berurutan tanpa mengubah urutan.
    deduplicated: list[str] = []
    for item in results:
        if (
            not deduplicated
            or normalize_text(item) != normalize_text(deduplicated[-1])
        ):
            deduplicated.append(item)

    return deduplicated


# ============================================================
# 4. EKSTRAKSI ENTITAS
# ============================================================

def extract_entities(text: str) -> dict[str, Any]:
    """
    Ekstraksi awal menggunakan aturan deterministik.
    Nama wilayah dan compartment harus divalidasi dengan data
    referensi proyek sebelum dipakai dalam query database.
    """
    entities: dict[str, Any] = {}
    normalized = normalize_text(text)

    # Tahun: contoh 2024 atau 2020
    year_match = re.search(
        r"\b(?:19|20)\d{2}\b",
        normalized,
    )
    if year_match:
        entities["year"] = int(year_match.group(0))

    # Growth period: P6, P12, P18, ..., PHI
    period_match = re.search(
        r"\b(?:p\d{1,3}|phi)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if period_match:
        entities["growth_period"] = period_match.group(0).upper()

    # Penanaman
    planting_match = re.search(
        r"\btahun\s+tanam\s+(?:tahun\s+)?((?:19|20)\d{2})\b",
        normalized,
    )
    if planting_match:
        entities["planting_year"] = int(planting_match.group(1))

    # Regional / area
    regional_match = re.search(
        r"\b(?:regional|wilayah|area)\s+([a-z0-9_-]+)\b",
        normalized,
    )
    if regional_match:
        entities["regional_candidate"] = regional_match.group(1).upper()

    # Compartment: bentuk umum, harus disesuaikan dengan ID asli.
    compartment_match = re.search(
        r"\bcompartment\s+([a-z0-9_-]+)\b"
        r"|\bkompartemen\s+([a-z0-9_-]+)\b",
        normalized,
    )
    if compartment_match:
        entities["compartment_candidate"] = (
            compartment_match.group(1)
            or compartment_match.group(2)
        ).upper()

        # Variabel target: satu pertanyaan boleh memiliki beberapa target
    targets = []

    if re.search(r"\b(dbh|diameter|diameter batang)\b", normalized):
        targets.append("diameter_cm")

    if re.search(r"\b(tinggi|height)\b", normalized):
        targets.append("height_m")

    if re.search(r"\b(volume|volume pohon)\b", normalized):
        targets.append("volume_tree_m3")

    if re.search(r"\b(kepadatan|densitas|tree density)\b", normalized):
        targets.append("tree_density")

    if targets:
        entities["targets"] = targets
        # Pertahankan kompatibilitas dengan kode lama yang memakai satu target
        entities["target"] = targets[0]

# ============================================================
# 5. KATALOG INTENT
# ============================================================

DEFAULT_INTENTS = [
    IntentSpec(
        name="explain_trajectory",
        description="Menjelaskan nilai trajectory atau kurva pertumbuhan.",
        examples=[
            "Apa itu nilai trajectory?",
            "Jelaskan trajectory pada pertumbuhan pohon.",
            "Apa arti kurva trajectory?",
            "Bagaimana memahami trajectory pertumbuhan?",
        ],
        handler_key="faq",
    ),
    IntentSpec(
        name="historical_growth_report",
        description="Mengambil laporan historis pertumbuhan pohon.",
        examples=[
            "Tampilkan laporan pertumbuhan pohon wilayah A tahun 2024.",
            "Cari data historis tinggi pohon pada tahun 2024.",
            "Saya ingin melihat laporan pertumbuhan regional.",
            "Tampilkan riwayat diameter dan volume pohon.",
        ],
        handler_key="historical_report",
    ),
    IntentSpec(
        name="download_growth_report",
        description="Mengunduh atau mengekspor laporan pertumbuhan.",
        examples=[
            "Bagaimana cara unduh laporan pertumbuhan?",
            "Download laporan pohon untuk wilayah A tahun 2024.",
            "Ekspor laporan historis ke PDF.",
            "Saya ingin mengunduh laporan pertumbuhan wilayah.",
        ],
        handler_key="download_report",
    ),
    IntentSpec(
        name="evaluate_random_forest",
        description="Menampilkan evaluasi model Random Forest.",
        examples=[
            "Bagaimana hasil evaluasi Random Forest?",
            "Seberapa akurat model random forest?",
            "Tampilkan MAE RMSE R2 dan MAPE Random Forest.",
            "Apa hasil pengujian model RF?",
        ],
        handler_key="faq",
    ),
    IntentSpec(
        name="evaluate_pinns",
        description="Menampilkan evaluasi model PINNs V2.",
        examples=[
            "Bagaimana hasil evaluasi PINNs V2?",
            "Tampilkan metrik test PINNs.",
            "Seberapa akurat model physics informed neural network?",
        ],
        handler_key="faq",
    ),
    IntentSpec(
        name="compare_models",
        description="Membandingkan dua model prediksi.",
        examples=[
            "Bandingkan Random Forest dengan PINNs V2.",
            "Model mana yang memiliki MAE lebih rendah?",
            "Apa perbedaan hasil evaluasi kedua model?",
        ],
        handler_key="faq",
    ),
]


# ============================================================
# 6. SEMANTIC MATCHER
# ============================================================

class LocalInputParser:
    def __init__(
        self,
        intents: Optional[list[IntentSpec]] = None,
        model_path: str = "models/distiluse-base-multilingual-cased-v1",
        min_similarity: float = 0.48,
        min_margin: float = 0.06,
        allow_model_download: bool = False,
    ):
        self.intents = intents or DEFAULT_INTENTS
        self.min_similarity = min_similarity
        self.min_margin = min_margin

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers belum terpasang. "
                "Install library di virtual environment proyek."
            ) from exc

        # False berarti hanya memuat model lokal.
        # True hanya boleh digunakan jika memang ingin mengunduh
        # model pada proses persiapan lingkungan.
        self.model = SentenceTransformer(
            model_path,
            local_files_only=not allow_model_download,
        )

        self.candidates: list[dict[str, Any]] = []
        for intent in self.intents:
            for example in intent.examples:
                self.candidates.append({
                    "intent": intent,
                    "text": example,
                })

        self.candidate_embeddings = self.model.encode(
            [item["text"] for item in self.candidates],
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

    def match_intent(self, text: str) -> ParsedQuery:
        from sentence_transformers import util

        normalized = normalize_text(text)
        result = ParsedQuery(
            original_text=text,
            normalized_text=normalized,
            entities=extract_entities(text),
        )

        if not normalized:
            result.status = "empty"
            return result

        query_embedding = self.model.encode(
            normalized,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

        scores = util.cos_sim(
            query_embedding,
            self.candidate_embeddings,
        )[0]

        ranked_indices = scores.argsort(descending=True).tolist()

        best_index = ranked_indices[0]
        best_candidate = self.candidates[best_index]
        best_score = float(scores[best_index])

        # Skor intent adalah skor contoh terbaik milik intent tersebut.
        intent_scores: dict[str, tuple[float, IntentSpec]] = {}

        for index, candidate in enumerate(self.candidates):
            intent = candidate["intent"]
            score = float(scores[index])

            previous = intent_scores.get(intent.name)
            if previous is None or score > previous[0]:
                intent_scores[intent.name] = (score, intent)

        ranked_intents = sorted(
            intent_scores.values(),
            key=lambda item: item[0],
            reverse=True,
        )

        best_score, best_intent = ranked_intents[0]
        second_score = (
            ranked_intents[1][0]
            if len(ranked_intents) > 1
            else 0.0
        )

        result.similarity = best_score
        result.second_similarity = second_score

        # Threshold dan margin adalah dua pemeriksaan berbeda.
        if best_score < max(self.min_similarity, best_intent.threshold):
            result.status = "unmatched"
            return result

        if best_score - second_score < self.min_margin:
            result.status = "ambiguous"
            return result

        result.intent = best_intent.name
        result.handler_key = best_intent.handler_key
        result.status = "matched"
        return result

    def parse(self, text: str) -> list[ParsedQuery]:
        sub_queries = split_queries(text)
        return [self.match_intent(query) for query in sub_queries]



    
# ============================================================
# 7. DISPATCHER MULTI-QUERY
# ============================================================

def dispatch_queries(
    parsed_queries: list[ParsedQuery],
    handlers: dict[str, Callable[[ParsedQuery], Any]],
) -> list[dict[str, Any]]:
    """
    Menjalankan setiap query secara berurutan.
    Handler harus berupa fungsi yang sudah tersedia dan tepercaya.
    """
    responses: list[dict[str, Any]] = []

    for index, parsed in enumerate(parsed_queries, start=1):
        item: dict[str, Any] = {
            "query_number": index,
            "query": parsed.original_text,
            "status": parsed.status,
            "intent": parsed.intent,
            "similarity": round(parsed.similarity, 4),
            "second_similarity": round(
                parsed.second_similarity, 4
            ),
            "entities": parsed.entities,
            "answer": None,
        }

        if parsed.status == "ambiguous":
            item["answer"] = (
                "Maksud pertanyaan ini belum cukup jelas. "
                "Mohon perjelas apakah Anda ingin melihat "
                "data historis, evaluasi model, atau mengunduh laporan."
            )
            responses.append(item)
            continue

        if parsed.status != "matched" or not parsed.handler_key:
            item["answer"] = (
                "Saya belum dapat menentukan fungsi yang tepat "
                "untuk pertanyaan ini."
            )
            responses.append(item)
            continue

        handler = handlers.get(parsed.handler_key)
        if handler is None:
            item["status"] = "handler_missing"
            item["answer"] = (
                f"Handler '{parsed.handler_key}' belum terhubung."
            )
            responses.append(item)
            continue

        try:
            # Handler lama dipanggil satu per satu.
            item["answer"] = handler(parsed)
            item["status"] = "success"
        except Exception:
            # Jangan tampilkan stack trace atau data sensitif
            # kepada pengguna akhir. Detail error dicatat
            # di log internal aplikasi.
            item["status"] = "handler_error"
            item["answer"] = (
                "Permintaan ini belum berhasil diproses. "
                "Silakan coba kembali atau persempit pertanyaan."
            )

        responses.append(item)

    return responses


