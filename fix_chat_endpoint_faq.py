from pathlib import Path
import re
import shutil

path = Path("api/main_new.py")
source = path.read_text(encoding="utf-8")

old = r'r"\b(?:ai/chat|endpoint chat|chat ai)\b",'
new = r'r"\b(?:ai\s*/\s*chat|endpoint\s+(?:ai\s+)?chat|chat\s+ai)\b",'

count = source.count(old)
if count != 1:
    raise SystemExit(
        f"STOP: Pola lama ditemukan {count} kali, bukan sekali. "
        "File tidak diubah."
    )

backup = path.with_name("main_new.before_chat_endpoint_faq_fix.py")
shutil.copy2(path, backup)

updated = source.replace(old, new, 1)
compile(updated, str(path), "exec")
path.write_text(updated, encoding="utf-8")

print("BERHASIL: Pola FAQ endpoint AI chat diperbaiki.")
print(f"Backup: {backup}")
print("Pemeriksaan sintaks Python: LULUS")
