# CURRENT CHECKPOINT: SIRIMBA

**Date:** 2026-11-10  
**Active Phase:** Phase 2 — Deployment Dependencies & Package Size Review  
**Completed Steps:**  
- [x] Step 1.1: Validate startup & /health endpoint  
- [x] Step 1.2: Safely Remove Unused PyTorch Code from `api/main_new.py`  
- [x] Step 1.3: Track & Commit ONNX Export and Validation Scripts (`b11432c`)  
- [x] Step 2.1: Create Minimal and Accurate `requirements_deploy.txt`  
**Upcoming Step:** Step 2.2: Verify Clean Deployment Environment Simulation

---

## Current System State

### 1. Backend Application (`api/main_new.py`)
- **Runtime Inference:** 100% ONNX Runtime (`onnxruntime.InferenceSession`) using `results/pinns/models/pinns_v2_growth_model.onnx` (`CPUExecutionProvider`).
- **PyTorch Status:** Obsolete PyTorch inference code (`import torch`, `import torch.nn`, `class TreeGrowthPINN`, `PINNS_MODEL_PATH`) has been safely removed from `api/main_new.py`. No active inference path requires PyTorch.
- **Random Forest:** Active using scikit-learn (`joblib`) models for DBH, Height, and Volume.
- **Dataset:** Active dataset `data/dataset_phi66_final.csv` (218,700 rows, 31 columns, P6 to P66 / PHI).

### 2. Deployment Requirements (`requirements_deploy.txt`)
- Created separate production requirements file with exactly 23 runtime packages.
- Completely excludes `torch`, `torchvision`, `transformers`, `sentence-transformers`, `safetensors`, and research packages.
- Includes `onnxruntime==1.31.1` and all required packages for FastAPI, scikit-learn unpickling, dataset handling (CSV/XLSX/XLS), PDF/PPTX generation, and LangGraph assistant.
- Deployment package size has not yet been measured. Actual installed size, dependency compatibility, and memory usage will be verified during Phase 2 Step 2.2.

### 3. Tracked ONNX Scripts (Commit: `b11432ca604ba5b165b7a1e5271b87aa9749d2fb`)
- `scripts/export_pinns_onnx.py` (ONNX export pipeline)
- `scripts/validate_pinns_onnx.py` (scaled numerical agreement validation, MAE $< 10^{-7}$)
- `scripts/validate_pinns_onnx_physical.py` (physical unit inverse-scaled validation)

### 4. Validation Status
- **Python Syntax Compilation:** PASSED
- **Static PyTorch Reference Search:** Clean (0 references in `api/main_new.py`)
- **Backend Startup:** SUCCESS (`FINAL BACKEND STARTUP : SUCCESS`)
- **GET /health:** HTTP 200 (`healthy`, `pinns_loaded: true`, `random_forest_loaded: true`, `dataset_loaded: true`)
- **GET /dashboard/trajectory:** HTTP 200 (`success: true`)
- **Direct ONNX Inference:** PASSED (DBH, Height, and non-negative Volume for age $\ge 18$ months verified)
- **Dependency Simulation:** PASSED (Backend boots and runs ONNX inference with PyTorch blocked from import. Note: this import simulation is not equivalent to a clean environment installation, which will be verified in Phase 2 Step 2.2)

### 5. Backup State
- Preserved backup: `api/main_new.backup_before_pytorch_cleanup_20261110_142200.py`
- Original model checkpoint preserved: `results/pinns/models/pinns_v2_growth_model.pt`
