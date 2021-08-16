
from pathlib import Path

Path.ls = lambda src: list(src.iterdir())

