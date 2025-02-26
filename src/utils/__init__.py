from pathlib import Path

class PATH:
    ROOT = Path(__file__).parents[2]
    
    DATA = ROOT / "data"
    RAW = DATA / "raw"
    PROCESSED = DATA / "processed"

    SRC = ROOT / "src"
    NOTEBOOKS = SRC / "notebooks"
    SCRIPTS = SRC / "scripts"
    