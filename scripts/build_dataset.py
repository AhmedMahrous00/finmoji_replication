import argparse, re, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

emoji_pattern = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF\u2700-\u27BF]"
)

url_re = re.compile(r"https?://\S+|www\.\S+")
cashtag_re = re.compile(r"[$#][A-Za-z0-9_]+")
multi_space = re.compile(r"\s+")

def has_emoji(s):
    return bool(emoji_pattern.search(s))

def strip_noncontent(s):
    s = url_re.sub(" ", s)
    s = cashtag_re.sub(" ", s)
    s = s.lower()
    s = multi_space.sub(" ", s).strip()
    return s

def remove_emojis(s):
    return emoji_pattern.sub("", s)

def extract_emojis(s):
    return " ".join(emoji_pattern.findall(s))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap-per-class", type=int, default=None)
    p.add_argument("--input-format", choices=["csv","parquet"], default="csv")
    args = p.parse_args()

    outdir = Path(args.outdir)
    (outdir / "splits").mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input}...")
    if args.input_format == "csv":
        df = pd.read_csv(args.input)
    else:
        df = pd.read_parquet(args.input, engine='pyarrow')
    print(f"Loaded {len(df)} rows")

    need_cols = {"text","sentiment","id"}
    if not need_cols.issubset(df.columns):
        missing = need_cols - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    print("Filtering for Bullish/Bearish sentiment...")
    df = df[df["sentiment"].isin(["Bullish","Bearish"])].copy()
    print(f"After sentiment filter: {len(df)} rows")
    
    print("Converting text to string and filtering for emoji presence...")
    df["text"] = df["text"].astype(str)
    tqdm.pandas(desc="Checking for emojis")
    df = df[df["text"].progress_map(has_emoji)].copy()
    print(f"After emoji filter: {len(df)} rows")

    g = df.groupby("sentiment", group_keys=False)
    counts = g.size().to_dict()
    n_min = min(counts.get("Bullish",0), counts.get("Bearish",0))
    target = n_min
    if args.cap_per_class is not None:
        target = min(target, args.cap_per_class)
    print(f"Balancing dataset: {target} samples per class...")
    df = pd.concat([g.get_group("Bullish").sample(target, random_state=args.seed),
                    g.get_group("Bearish").sample(target, random_state=args.seed)], axis=0)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"Balanced dataset: {len(df)} total rows")

    print("Processing text variants...")
    tqdm.pandas(desc="Stripping non-content")
    clean = df["text"].progress_map(strip_noncontent)
    text_emoji = clean
    tqdm.pandas(desc="Removing emojis")
    text_only = clean.progress_map(remove_emojis)
    tqdm.pandas(desc="Extracting emojis")
    emoji_only = df["text"].progress_map(extract_emojis)

    out = pd.DataFrame({
        "post_id": df["id"],
        "label": df["sentiment"].str.lower(),
        "text_raw": df["text"],
        "text_only": text_only,
        "emoji_only": emoji_only,
        "text_emoji": text_emoji
    })

    test_ids = train_test_split(out["post_id"], test_size=0.10, stratify=out["label"], random_state=args.seed)[1]
    remain = out[~out["post_id"].isin(test_ids)]
    y_remain = remain["label"]
    train_ids, val_ids = train_test_split(remain["post_id"], test_size=0.10/0.90, stratify=y_remain, random_state=args.seed)

    out_text_only = out[["post_id","label","text_only"]].rename(columns={"text_only":"text"})
    out_emoji_only = out[["post_id","label","emoji_only"]].rename(columns={"emoji_only":"text"})
    out_text_emoji = out[["post_id","label","text_emoji"]].rename(columns={"text_emoji":"text"})

    print("Writing output files...")
    out_text_only.to_csv(outdir/"text_only.csv", index=False)
    out_emoji_only.to_csv(outdir/"emoji_only.csv", index=False)
    out_text_emoji.to_csv(outdir/"text_emoji.csv", index=False)

    Path(outdir/"splits"/"train_ids.txt").write_text("\n".join(map(str, train_ids)))
    Path(outdir/"splits"/"val_ids.txt").write_text("\n".join(map(str, val_ids)))
    Path(outdir/"splits"/"test_ids.txt").write_text("\n".join(map(str, test_ids)))
    print(f"Done! Output written to {outdir}/")

if __name__ == "__main__":
    main()
