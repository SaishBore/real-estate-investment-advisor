"""
Write requirements.txt using the exact package versions installed on YOUR computer.
Saved models only load reliably with the same library versions, so pinning matters.
Run from the project root:  python make_requirements.py
"""
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ["pandas", "numpy", "scikit-learn", "xgboost", "streamlit", "mlflow",
            "joblib", "matplotlib", "seaborn", "jinja2"]

lines = []
for name in PACKAGES:
    try:
        lines.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        print(f"Not installed, skipped: {name}")

with open("requirements.txt", "w") as f:
    f.write("\n".join(lines) + "\n")

print("Wrote requirements.txt:")
print("\n".join(lines))