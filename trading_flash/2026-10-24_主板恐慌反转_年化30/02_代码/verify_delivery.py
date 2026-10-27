from pathlib import Path
import runpy
import sys

AUDIT = Path(__file__).resolve().parents[3] / "trading_hithink" / "flash 审计修复"
if not (AUDIT / "verify_audit.py").is_file():
    raise FileNotFoundError(AUDIT / "verify_audit.py")
sys.path.insert(0, str(AUDIT))
runpy.run_path(str(AUDIT / "verify_audit.py"), run_name="__main__")
