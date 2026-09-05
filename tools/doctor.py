"""Environment report for the book's code samples.

Run via ``make doctor``. Prints the interpreter and platform actually in use,
then checks each dependency and flags the combinations that are known to break.
Most "it doesn't run for me" reports come down to the wrong interpreter being
active, which the first two lines of output make obvious.
"""

import importlib
import platform
import sys

# (import name, pip name, which requirements file provides it)
PACKAGES = [
    ("numpy", "numpy", "requirements.txt"),
    ("matplotlib", "matplotlib", "requirements.txt"),
    ("pandas", "pandas", "requirements.txt"),
    ("gymnasium", "gymnasium", "requirements.txt"),
    ("torch", "torch", "requirements-deep.txt"),
    ("cv2", "opencv-python", "requirements-deep.txt"),
    ("ale_py", "ale-py", "requirements-atari.txt (optional)"),
    # Chapter 7 only, and optional the same way Atari is: chapters 1-6 have no
    # use for it, and two of Chapter 7's three targets run without it.
    ("transformers", "transformers", "requirements-llm.txt (optional)"),
    ("peft", "peft", "requirements-llm.txt (optional)"),
]


def _version(module):
    return getattr(module, "__version__", "unknown")


def main():
    py = sys.version_info
    machine = platform.machine()
    is_intel_mac = sys.platform == "darwin" and machine == "x86_64"

    print("Environment report")
    print("==================")
    print(f"python      : {platform.python_version()}")
    print(f"executable  : {sys.executable}")
    print(f"platform    : {platform.platform()}")
    print(f"machine     : {machine}")
    print()

    print("Packages")
    print("--------")
    found = {}
    missing = []
    for import_name, pip_name, source in PACKAGES:
        try:
            module = importlib.import_module(import_name)
        except ImportError:
            print(f"  {pip_name:<16} MISSING   (from {source})")
            missing.append((pip_name, source))
        else:
            found[import_name] = _version(module)
            print(f"  {pip_name:<16} {found[import_name]}")
    print()

    warnings = []

    if py < (3, 10):
        warnings.append(
            f"Python {platform.python_version()} is below the 3.10 minimum "
            "required by gymnasium >= 1.0 and torch >= 2.0."
        )

    if is_intel_mac:
        warnings.append(
            "Intel macOS detected. PyTorch ships no x86_64 macOS wheels past "
            "2.2.x, so this platform is capped at torch 2.2.2 and Python 3.12. "
            "Newer Python versions here will fail to install torch at all."
        )

    if "torch" not in found and py >= (3, 13) and is_intel_mac:
        warnings.append(
            "torch is missing and cannot be installed on Intel macOS under "
            f"Python {platform.python_version()}. Recreate the virtualenv with "
            "Python 3.12."
        )

    numpy_version = found.get("numpy")
    if numpy_version and "torch" in found:
        numpy_major = int(numpy_version.split(".")[0])
        torch_minor = tuple(int(p) for p in found["torch"].split(".")[:2])
        if numpy_major >= 2 and torch_minor < (2, 3):
            warnings.append(
                f"torch {found['torch']} was built against NumPy 1.x but "
                f"NumPy {numpy_version} is installed. Expect a "
                '"compiled using NumPy 1.x" error at import time.'
            )

    if warnings:
        print("Warnings")
        print("--------")
        for warning in warnings:
            print(f"  ! {warning}")
        print()

    if missing:
        print("Next steps")
        print("----------")
        files = {source for _, source in missing}
        if any(f.startswith("requirements.txt") for f in files):
            print("  make install")
        if any(f.startswith("requirements-deep") for f in files):
            print("  make install-full")
        if any(f.startswith("requirements-atari") for f in files):
            print("  make install-atari      # optional, Chapter 3 Atari only")
        if any(f.startswith("requirements-llm") for f in files):
            print("  make install-llm        # optional, Chapter 7 only")
        print()
        print("  If those install into the wrong interpreter, activate your")
        print("  virtualenv first, or pass one explicitly:")
        print("      make install-full PYTHON=/path/to/.venv/bin/python")
    elif warnings:
        print("All packages present, but see the warnings above.")
    else:
        print("All good.")


if __name__ == "__main__":
    # Always exits 0: this is a report, not a gate. A non-zero exit would make
    # `make doctor` print its own "Error 1" on top of the diagnosis.
    main()
