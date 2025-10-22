import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_emoji_count_sentiment(input_csv_path, output_dir):
    print(f"Loading {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    df["label"] = df["label"].str.lower()

    print("Calculating unique emoji counts per post...")
    df["unique_emoji_count"] = df["text"].apply(
        lambda x: len(set(str(x).split())) if pd.notna(x) and str(x).strip() else 0
    )

    df = df[df["unique_emoji_count"] > 0].copy()

    df["emoji_count_group"] = df["unique_emoji_count"].apply(lambda x: min(x, 10))
    df.loc[df["emoji_count_group"] == 10, "emoji_count_group"] = "10+"

    print("Calculating sentiment percentages and frequencies...")
    grouped_data = df.groupby("emoji_count_group").agg(
        total_posts=("post_id", "count"),
        bullish_posts=("label", lambda x: (x == "bullish").sum()),
        bearish_posts=("label", lambda x: (x == "bearish").sum())
    ).reset_index()

    grouped_data["bullish_percentage"] = (grouped_data["bullish_posts"] / grouped_data["total_posts"]) * 100
    grouped_data["bearish_percentage"] = (grouped_data["bearish_posts"] / grouped_data["total_posts"]) * 100

    grouped_data["emoji_count_group"] = grouped_data["emoji_count_group"].astype(str)
    grouped_data.loc[grouped_data["emoji_count_group"] == "10", "emoji_count_group"] = "10+"
    
    sort_order = [str(i) for i in range(1, 10)] + ["10+"]
    grouped_data["emoji_count_group"] = pd.Categorical(grouped_data["emoji_count_group"], categories=sort_order, ordered=True)
    grouped_data = grouped_data.sort_values("emoji_count_group")

    print("\nAggregated Data for Chart:")
    print(grouped_data)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    bar_width = 0.35
    x = range(len(grouped_data))

    ax1.bar([i - bar_width/2 for i in x], grouped_data["bullish_percentage"], bar_width, 
            label="Bullish", color="forestgreen")
    ax1.bar([i + bar_width/2 for i in x], grouped_data["bearish_percentage"], bar_width, 
            label="Bearish", color="red")

    ax1.set_xlabel("Emoji Count")
    ax1.set_ylabel("Sentiment Percentage", color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_ylim(0, 70)

    ax2 = ax1.twinx()
    ax2.plot(x, grouped_data["total_posts"], color="royalblue", marker="o", linestyle="-", 
             label="Frequency", linewidth=2, markersize=8)
    ax2.set_ylabel("Frequency", color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")
    
    for i, txt in enumerate(grouped_data["total_posts"]):
        ax2.annotate(f"{txt:,}", (x[i], grouped_data["total_posts"].iloc[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    color='royalblue', fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(grouped_data["emoji_count_group"])

    plt.title("Emoji Count versus Sentiment and Frequency", fontsize=14, fontweight='bold')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.tight_layout()

    output_path = Path(output_dir) / "emoji_count_sentiment_frequency.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nChart saved to {output_path}")
    plt.show()

    csv_output_path = Path(output_dir) / "emoji_count_sentiment_data.csv"
    grouped_data.to_csv(csv_output_path, index=False)
    print(f"Data saved to {csv_output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to emoji_only.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    analyze_emoji_count_sentiment(args.input, args.output_dir)
