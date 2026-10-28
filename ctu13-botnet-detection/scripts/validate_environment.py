"""
scripts/validate_environment.py
Check that all required packages are installed and importable.
"""
import sys
from pathlib import Path

REQUIRED = [
    ("pandas",         "pandas"),
    ("numpy",          "numpy"),
    ("sklearn",        "scikit-learn"),
    ("matplotlib",     "matplotlib"),
    ("seaborn",        "seaborn"),
    ("yaml",           "PyYAML"),
    ("joblib",         "joblib"),
    ("scipy",          "scipy"),
]
OPTIONAL = [
    ("xgboost",        "xgboost"),
    ("lightgbm",       "lightgbm"),
    ("imblearn",       "imbalanced-learn"),
]

ok = True
print("=" * 50)
print("CTU-13 Botnet Detection — Environment Check")
print("=" * 50)
print(f"Python: {sys.version}")
print()

print("Required packages:")
for module, pkg in REQUIRED:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        print(f"  ✓  {pkg:<22} {ver}")
    except ImportError:
        print(f"  ✗  {pkg:<22} NOT INSTALLED")
        ok = False

print()
print("Optional packages:")
for module, pkg in OPTIONAL:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        print(f"  ✓  {pkg:<22} {ver}")
    except ImportError:
        print(f"  -  {pkg:<22} not installed (optional)")

print()
if ok:
    print("✓  All required packages are available.")
    sys.exit(0)
else:
    print("✗  Some required packages are missing. Run:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
