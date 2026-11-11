import json
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

try:
    from api.ai_tools import sirimba_tools
except ModuleNotFoundError:
    from ai_tools import sirimba_tools

try:
    from api.llm_provider import invoke_llm
except ModuleNotFoundError:
    from llm_provider import invoke_llm


class AIState(TypedDict):
    question: str
    intent: str
    answer: str


TOOLS_BY_INTENT = {
    "general": "get_growth_data",
    "comparison": "compare_growth_data",
    "prediction": "predict_tree_growth",
}


def classify_question(state: AIState) -> dict:
    question = state["question"]
    question_lower = question.lower()

    if any(term in question_lower for term in (
        "apa itu", "jelaskan", "definisi", "pengertian",
    )):
        return {"intent": "general"}

    if any(term in question_lower for term in (
        "bandingkan", "perbandingan", "compare",
        "komparasi", "mana lebih baik",
    )):
        return {"intent": "comparison"}

    if any(term in question_lower for term in (
        "prediksi", "predict", "prediction",
        "ramalkan", "estimasi pertumbuhan",
    )):
        return {"intent": "prediction"}

    if any(term in question_lower for term in (
        "rata-rata", "rata rata", "average", "mean",
        "jumlah", "total", "maksimum", "minimum",
        "tertinggi", "terendah", "median",
        "berapa banyak", "statistik",
    )):
        return {"intent": "general"}

    prompt = (
        "Klasifikasikan pertanyaan ke satu kategori: "
        "comparison, prediction, atau general.\n"
        "Jawab hanya satu kata.\n\n"
        f"Pertanyaan: {question}"
    )

    response = invoke_llm(prompt)

    if response:
        intent = response.strip().lower()
        if intent in ("prediction", "comparison", "general"):
            return {"intent": intent}

    return {"intent": "general"}


def handle_question(state: AIState) -> dict:
    intent = state["intent"]
    question = state["question"]

    tool_name = TOOLS_BY_INTENT.get(intent, "get_growth_data")
    selected_tool = next(
        (tool for tool in sirimba_tools if tool.name == tool_name),
        None,
    )

    if selected_tool is None:
        return {"answer": f"Tool tidak ditemukan: {tool_name}"}

    try:
        result = selected_tool.invoke({"question": question})

        if isinstance(result, str):
            raw_result = result
        else:
            raw_result = json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            )

        if isinstance(result, dict) and result.get("success") is False:
            return {"answer": raw_result}

        prompt = (
            "Kamu adalah AI Assistant SIRIMBA.\n"
            "Jawab pertanyaan pengguna dalam bahasa Indonesia "
            "yang jelas dan ringkas.\n"
            "Gunakan hanya informasi dari hasil tool.\n"
            "Jangan mengarang angka, fakta, atau hasil prediksi.\n"
            "Jika informasi tidak tersedia, katakan bahwa "
            "informasi tersebut tidak tersedia.\n\n"
            f"Pertanyaan pengguna:\n{question}\n\n"
            f"Hasil tool:\n{raw_result}"
        )

        response = invoke_llm(prompt)

        if response:
            return {"answer": response}

        # Fallback: hasil analitik tetap tersedia tanpa LLM.
        return {"answer": raw_result}

    except Exception as exc:
        return {
            "answer": (
                f"Terjadi kesalahan saat menjalankan "
                f"{tool_name}: {type(exc).__name__}: {exc}"
            )
        }


def build_ai_graph():
    graph = StateGraph(AIState)

    graph.add_node("classify", classify_question)
    graph.add_node("handle", handle_question)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "handle")
    graph.add_edge("handle", END)

    return graph.compile()


ai_graph = build_ai_graph()
