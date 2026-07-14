"""Repository launcher for RoadScope-N5 image inference."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from roadscope_n5.inference import main


if __name__ == "__main__":
    main()
