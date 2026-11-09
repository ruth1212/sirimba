from langchain_core.tools import tool


def _get_backend_module():
    import sys

    if "main_new" in sys.modules:
        return sys.modules["main_new"]

    if "api.main_new" in sys.modules:
        return sys.modules["api.main_new"]

    try:
        import main_new
        return main_new
    except ModuleNotFoundError:
        import api.main_new
        return api.main_new


@tool
def get_growth_data(question: str) -> dict:
    """Mengambil informasi dan statistik pertumbuhan pohon dari dataset SIRIMBA."""
    backend = _get_backend_module()
    return backend.query_growth_data(question)


@tool
def compare_growth_data(question: str) -> dict:
    """Membandingkan pertumbuhan pohon berdasarkan regional, compartment, species, atau clone."""
    backend = _get_backend_module()
    result = backend.compare_growth_groups(question)

    if result is None:
        return {
            "success": False,
            "message": "Pertanyaan belum dapat diproses sebagai perbandingan."
        }

    return result


@tool
def predict_tree_growth(question: str) -> dict:
    """Menghasilkan prediksi pertumbuhan pohon menggunakan model Python yang tersedia."""
    backend = _get_backend_module()
    result = backend.answer_prediction_question(question)

    if result is None:
        return {
            "success": False,
            "message": "Pertanyaan belum dapat diproses sebagai prediksi."
        }

    return result


sirimba_tools = [
    get_growth_data,
    compare_growth_data,
    predict_tree_growth,
]
