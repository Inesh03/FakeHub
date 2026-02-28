import pandas as pd
import os

BASE = "data/datasets_full.csv"

folders = {
    "genuine_accounts.csv":     0,  # Human
    "fake_followers.csv":       1,  # Bot
    "social_spambots_1.csv":    1,  # Bot
    "social_spambots_2.csv":    1,  # Bot
    "social_spambots_3.csv":    1,  # Bot
    "traditional_spambots_1.csv": 1,
    "traditional_spambots_2.csv": 1,
    "traditional_spambots_3.csv": 1,
    "traditional_spambots_4.csv": 1,
}

dfs = []

for folder, label in folders.items():
    path = os.path.join(BASE, folder, "tweets.csv")
    if not os.path.exists(path):
        print(f"⚠️  Skipping (not found): {path}")
        continue
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", usecols=["text"])
    df["label"] = label
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]
    print(f"{'🟢 Human' if label == 0 else '🔴 Bot   '} | {folder:<35} | {len(df):>7} rows")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Cap at 100k for fast training
if len(combined) > 100000:
    combined = combined.sample(100000, random_state=42).reset_index(drop=True)
    print(f"\n⚡ Capped to 100,000 rows for faster training")

output = "data/twibot22_sample.csv"
combined.to_csv(output, index=False)

print(f"\n{'='*55}")
print(f"✅ Saved → {output}")
print(f"Total rows : {len(combined):,}")
print(combined["label"].value_counts().rename({0: "Human 🟢", 1: "Bot 🔴"}))
