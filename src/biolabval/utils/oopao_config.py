import os

os.environ["OOPAO_PRECISION"] = "32"  # [bits] "32" or "64"
os.environ["OOPAO_BACKEND"] = (
    "cpu"  # "cpu" or "cuda" or "auto" (auto will use cuda if available, otherwise cpu)
)
