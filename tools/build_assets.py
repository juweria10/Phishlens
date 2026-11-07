"""Build the small lookup tables the URL scanner needs (run once, from the project root).

    python tools/build_assets.py "path/to/cleaned_URL_Dataset.csv"

Writes assets/tld_prob.json and assets/char_prob.json so the app does not need the 9 MB dataset.
"""
import collections
import json
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("EDA result/cleaned_URL_Dataset.csv")
df = pd.read_csv(src, encoding="latin1").dropna(subset=["URL", "label"]).drop_duplicates("URL")

# TLDLegitimateProb is a fixed value per TLD in the dataset, so one row per TLD is enough.
tld = (
    df.dropna(subset=["TLD", "TLDLegitimateProb"])
    .groupby(df["TLD"].str.lower())["TLDLegitimateProb"]
    .first()
    .to_dict()
)

# Character frequencies across legitimate URLs, used to rebuild URLCharProb.
legit = df[df["label"] == 1]
counts = collections.Counter("".join(legit["URL"]))
total = sum(counts.values())
chars = {c: n / total for c, n in counts.items()}

out = Path(__file__).resolve().parent.parent / "assets"
out.mkdir(exist_ok=True)
json.dump(tld, open(out / "tld_prob.json", "w"), indent=0)
json.dump(chars, open(out / "char_prob.json", "w"), indent=0)
print(f"{len(tld)} TLDs and {len(chars)} characters written to {out}")
