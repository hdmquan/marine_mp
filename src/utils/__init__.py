from pathlib import Path

class PATH:
    ROOT = Path(__file__).parents[2]
    
    DATA = ROOT / "data"
    RAW = DATA / "raw"
    PROCESSED = DATA / "processed"
    PREDICTIONS = PROCESSED / "predictions"

    SRC = ROOT / "src"
    NOTEBOOKS = SRC / "notebooks"
    SCRIPTS = SRC / "scripts"

    WEIGHTS = ROOT / "weights"