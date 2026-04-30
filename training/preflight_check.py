"""
SalesPath — Pre-flight Dependency Check
Run at the start of training to catch version mismatches early.
"""
import sys
import importlib

REQUIRED_PACKAGES = {
    "torch": "2.0.0",
    "transformers": "4.44.2",
    "trl": "0.11.0",
    "peft": "0.11.1",
    "datasets": "2.0.0",
    "fastapi": "0.100.0",
    "httpx": "0.24.0",
    "openenv": None,
    "accelerate": "0.25.0",
}

all_ok = True

print("=" * 60)
print("SalesPath Pre-flight Check")
print("=" * 60)

# Python version
print(f"Python: {sys.version}")
if sys.version_info < (3, 10):
    print("  WARNING: Python >= 3.10 recommended")
    all_ok = False

# CUDA availability
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
except Exception as e:
    print(f"PyTorch: ERROR — {e}")
    all_ok = False

# Check each package
for pkg_name, min_version in REQUIRED_PACKAGES.items():
    try:
        mod = importlib.import_module(pkg_name)
        ver = getattr(mod, "__version__", "unknown")
        status = f"{ver}"
        if min_version:
            from packaging import version
            if version.parse(ver) < version.parse(min_version):
                status += f" (needs >= {min_version}) ⚠️"
                all_ok = False
            else:
                status += " ✅"
        else:
            status += " ✅"
        print(f"{pkg_name}: {status}")
    except ImportError:
        print(f"{pkg_name}: NOT FOUND ❌")
        all_ok = False
    except Exception as e:
        print(f"{pkg_name}: ERROR — {e} ❌")
        all_ok = False

print("=" * 60)
if all_ok:
    print("All checks passed ✅")
else:
    print("Some checks failed ⚠️ — training may still work")
print("=" * 60)

sys.exit(0 if all_ok else 1)
