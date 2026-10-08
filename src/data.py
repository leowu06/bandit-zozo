"""Load the ZOZOTOWN logs into one clean DataFrame."""
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = {
    10: "zozo_noContext_10items.csv",
    20: "zozo_Context_20items.csv",
    80: "zozo_Context_80items.csv.zip",
}

USER_COLS = ["user_feature_0", "user_feature_1", "user_feature_2", "user_feature_3"]


def load(n_items=80):
    """Return the log for the 10, 20 or 80 item file, sorted by time.

    Changes made to the raw file:
      * the unnamed row index column is dropped
      * timestamp becomes a pandas datetime (UTC)
      * item_id is mapped to 0..K-1 so it can index numpy arrays
        (the original id is kept as item_id_raw)
      * every hashed user feature becomes a small integer code 0..L-1
    """
    path = DATA_DIR / FILES[n_items]
    if path.suffix == ".zip":
        # the zip also contains a __MACOSX file, so open the csv by name
        with zipfile.ZipFile(path) as z:
            with z.open(path.stem) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)

    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    df["item_id_raw"] = df["item_id"]
    df["item_id"] = df["item_id"].astype("category").cat.codes.astype(int)

    for c in USER_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category").cat.codes.astype(int)

    return df


def user_cols(df):
    """User feature columns present in this file (none for the 10 item file)."""
    return [c for c in USER_COLS if c in df.columns]
