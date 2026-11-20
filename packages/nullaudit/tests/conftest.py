# Copyright 2026 The HandReel authors. SPDX-License-Identifier: Apache-2.0
"""CPU only: the GPU is hidden before any test module imports a simulator."""

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src"), str(HERE)]
