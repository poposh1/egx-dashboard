"""التحديث اليومي: 04 ثم 05. لو أي خطوة فشلت بنوقف (exit code != 0) عشان GitHub Actions ما يعملش commit لداتا ناقصة."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

for step in ["update_data.py", "train_and_predict.py"]:
    print(f"\n===== {step} =====", flush=True)
    result = subprocess.run([sys.executable, str(HERE / step)])
    if result.returncode != 0:
        sys.exit(f"{step} failed (exit code {result.returncode}) - stopping.")

print("\nDaily update finished OK.")
