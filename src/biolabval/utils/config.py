# %%

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Paths:

    root_dir: Path = (
        Path(__file__).resolve().parents[3]
    )  # root directory of the project
    data_dir: Path = root_dir / "data"


@dataclass
class Config:

    paths: Paths = field(default_factory=Paths)
