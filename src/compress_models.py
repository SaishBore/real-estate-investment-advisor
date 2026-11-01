"""
Shrink the saved model files so they are comfortable to upload to GitHub
(GitHub warns above 50 MB per file and blocks files above 100 MB).
The models behave exactly the same; only the file size changes.
Run from the project root:  python compress_models.py
"""
import os

import joblib

for name in ["best_classifier.joblib", "best_regressor.joblib"]:
    path = os.path.join("models", name)
    if not os.path.exists(path):
        print(f"Skipped (not found): {path}")
        continue
    before = os.path.getsize(path) / 1_000_000
    model = joblib.load(path)
    joblib.dump(model, path + ".tmp", compress=3)
    os.replace(path + ".tmp", path)          # only replace the original once the new file is written
    after = os.path.getsize(path) / 1_000_000
    print(f"{name}: {before:.1f} MB -> {after:.1f} MB")