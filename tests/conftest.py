import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CH2 = ROOT / "src" / "part_1_foundations" / "ch02_fundamentals"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CH2) not in sys.path:
    sys.path.insert(0, str(CH2))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: end-to-end runs (notebook execution, training loops). "
        "Skipped by `make test`, included by `make test-all`.",
    )
