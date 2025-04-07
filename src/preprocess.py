# src/preprocess.py

import pandas as pd
from pathlib import Path

def load_bbc_dataset(data_dir="data/bbc"):
    texts, labels = [], []
    base_path = Path(data_dir)

    for category in base_path.iterdir():
        if category.is_dir():
            for file in category.glob("*.txt"):
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read().strip())
                    labels.append(category.name)

    df = pd.DataFrame({"text": texts, "category": labels})
    return df