# DEVELOPMENT LOG: SIRIMBA

## [2026-11-10] Phase 1 — Step 1.2: Safely Remove Unused PyTorch Code from api/main_new.py

### Objective
Remove obsolete PyTorch inference code from `api/main_new.py` without modifying active application behavior or affecting Random Forest and other API features.

### Pre-Edit Analysis
- `import torch` and `import torch.nn as nn` were only used by the obsolete `class TreeGrowthPINN(nn.Module)`.
- `load_pinns()` now loads the model using `onnxruntime.InferenceSession` on `pinns_v2_growth_model.onnx`.
- `predict_pinns()` passes `np.float32` arrays directly to `pinns_model.run(...)` and uses scikit-learn `target_scaler` for inverse scaling.
- `PINNS_MODEL_PATH` pointed to the `.pt` file and was unused in `api/main_new.py`.

### Backup Created
- `api/main_new.backup_before_pytorch_cleanup_20261110_142200.py` (372,330 bytes)
- All prior backups and `.pt` checkpoints remain intact.

### Code Removed from `api/main_new.py`
1. `import torch`
2. `import torch.nn as nn`
3. `PINNS_MODEL_PATH = PROJECT_ROOT / "results" / "pinns" / "models" / "pinns_v2_growth_model.pt"`
4. `class TreeGrowthPINN(nn.Module)` (lines 647–715)

### Code Preserved
- `import onnxruntime as ort`
- `PINNS_ONNX_MODEL_PATH`
- `pinns_model` as the ONNX `InferenceSession` object
- `feature_scaler` and `target_scaler`
- `load_pinns()` and `predict_pinns()`
- All Random Forest regressors and data processing workflows
- Volume threshold rule (predictions available only for `age_months >= 18`)

### Verification Suite Executed
1. `py_compile.compile('api/main_new.py')` $\rightarrow$ PASSED.
2. Static search for `import torch`, `from torch`, `torch.`, `nn.Module`, `TreeGrowthPINN`, `PINNS_MODEL_PATH` $\rightarrow$ 0 matches found (CLEAN).
3. `import api.main_new` $\rightarrow$ SUCCESS.
4. Full application startup $\rightarrow$ SUCCESS (`pinns_loaded: True`, `dataset_loaded: True`, `random_forest_loaded: True`).
5. `GET /health` endpoint $\rightarrow$ HTTP 200 OK.
6. `GET /dashboard/trajectory` endpoint $\rightarrow$ HTTP 200 OK.
7. Direct smoke test with `predict_pinns()` on real dataset rows $\rightarrow$ PASSED (DBH, Height, and non-negative Volume for mature cohorts).

### Verdict
**PASSED** — No active inference path in `api/main_new.py` requires PyTorch.


## [2026-11-10] Phase 1 — Step 1.3: Track and Commit ONNX Export and Validation Scripts

### Objective
Track and commit the three standalone ONNX export and validation scripts to the SIRIMBA Git repository.

### Initial Status
- `scripts/export_pinns_onnx.py` — Untracked (`??`)
- `scripts/validate_pinns_onnx.py` — Untracked (`??`)
- `scripts/validate_pinns_onnx_physical.py` — Untracked (`??`)

### Review Findings
- All three scripts resolve paths dynamically relative to repository root (`Path(__file__).resolve().parents[1]`).
- No hardcoded local machine paths.
- No secrets, API keys, or embedded dataset/model binaries.
- `.gitignore` allows the Python scripts under `scripts/`.
- Full execution test:
  - `validate_pinns_onnx.py` on 43,740 test rows $\rightarrow$ Numerical agreement PASSED (MAE difference $< 10^{-7}$).
  - `validate_pinns_onnx_physical.py` on 43,740 test rows $\rightarrow$ Physical unit agreement PASSED.

### Staged Files
- `scripts/export_pinns_onnx.py` (167 lines)
- `scripts/validate_pinns_onnx.py` (118 lines)
- `scripts/validate_pinns_onnx_physical.py` (109 lines)

### Commit
- **Commit Hash:** `b11432ca604ba5b165b7a1e5271b87aa9749d2fb` (`b11432c`)
- **Commit Message:** `chore(pinns): track ONNX export and validation scripts`
- **Scope:** Exactly 3 files tracked, no unrelated files staged or committed.

### Final Git Status
- Active Branch: `master`
- Tracking: `origin/master` (1 commit ahead locally, not pushed)
- Working Tree: Unstaged changes in `api/main_new.py`, documentation untracked, local log files untracked.


## [2026-11-10] Phase 2 — Step 2.1: Create Minimal and Accurate requirements_deploy.txt

### Objective
Create a separate deployment requirements file (`requirements_deploy.txt`) containing only the dependencies needed to run the active SIRIMBA backend in production, completely excluding PyTorch, Transformers, and research-only packages.

### Active Backend Modules Inspected
- `api/main_new.py`
- `api/dataset_standardizer.py`
- `api/report_generator.py`
- `api/ai_graph.py`
- `api/ai_tools.py`
- `api/ai_assistant_engine.py`

### Newly Created File
- `requirements_deploy.txt` (23 pinned production packages)

### Excluded Packages and Rationale
1. `torch==2.14.0`, `torchvision`: Excluded ($\approx 1.5\text{ GB}$). Inference fully migrated to ONNX Runtime.
2. `transformers==5.20.2`, `sentence-transformers==6.1.0`: Excluded ($\approx 500\text{ MB}$). HuggingFace models/embeddings not used at API runtime.
3. `safetensors==0.8.0`, `tokenizers==0.23.3`, `huggingface_hub==1.33.0`: Excluded. Transitive dependencies of transformers.
4. `sympy==1.14.0`, `mpmath==1.3.0`: Excluded. Symbolic PDE math only used in offline research scripts.
5. `statsmodels==0.15.0`, `patsy==1.0.3`, `formulaic==1.2.2`: Excluded. Statistical analysis packages used only in offline exploration scripts.
6. `onnx==1.23.2`: Excluded. Model exporter only used in `export_pinns_onnx.py`; runtime only needs `onnxruntime`.
7. `watchfiles==1.3.0`, `rich==15.0.0`, `Pygments==2.21.0`: Excluded. Development/CLI formatting tools.

### Validation Results
- Python AST import extraction confirmed all third-party imports in `api/` are covered.
- No duplicate package declarations (23 unique packages).
- Simulation with PyTorch blocked from `sys.meta_path` passed startup and direct ONNX predictions without errors.
- `requirements_current.txt` was preserved untouched.

### Remaining Deployment Risks
- Local Ollama dependency in `ai_graph.py` requires addressing in Step 3.1 for container environments where `localhost:11434` is unavailable.
- Ephemeral filesystem persistence for uploaded datasets in `data/dataset_staging/`.

### Verdict
**PASSED** — `requirements_deploy.txt` created and validated accurately.
