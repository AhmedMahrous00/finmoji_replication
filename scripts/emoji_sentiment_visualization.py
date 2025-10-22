import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure matplotlib for better emoji support
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Apple Color Emoji', 'Segoe UI Emoji']

def create_emoji_sentiment_bar_chart(input_csv_path, output_dir, top_k=50):
    print(f"Loading emoji sentiment data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    if 'emoji' not in df.columns or 'bullish_prop' not in df.columns or 'bearish_prop' not in df.columns:
        print("Error: Required columns not found. Expected: emoji, bullish_prop, bearish_prop")
        return
    
    df = df.head(top_k)
    
    df = df.sort_values('bearish_prop', ascending=False)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    x_pos = np.arange(len(df))
    bar_width = 0.8
    
    ax.bar(x_pos, df['bearish_prop'] * 100, bar_width, color='red', alpha=0.7, label='Bearish')
    ax.bar(x_pos, df['bullish_prop'] * 100, bar_width, bottom=df['bearish_prop'] * 100, 
           color='green', alpha=0.7, label='Bullish')
    
    ax.set_xlabel('Individual Emojis', fontsize=12)
    ax.set_ylabel('Proportion of Posts (%)', fontsize=12)
    ax.set_title('Sentiment Score of Individual Emojis', fontsize=16, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['emoji'], rotation=90, fontsize=10)
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "emoji_sentiment_bar_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Emoji sentiment bar chart saved to {output_path}")
    plt.show()
    
    return df

def create_emoji_distribution_histogram(input_csv_path, output_dir):
    print(f"Loading emoji data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    df["unique_emoji_count"] = df["text"].apply(
        lambda x: len(set(str(x).split())) if pd.notna(x) and str(x).strip() else 0
    )
    
    df["total_emoji_count"] = df["text"].apply(
        lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() else 0
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    bins = list(range(0, 11))
    bins.append(20)
    labels = [str(i) for i in range(0, 10)] + ['10+']
    
    df["emoji_count_group"] = pd.cut(df["unique_emoji_count"], bins=bins, labels=labels, right=False)
    df["total_emoji_count_group"] = pd.cut(df["total_emoji_count"], bins=bins, labels=labels, right=False)
    
    unique_counts = df["emoji_count_group"].value_counts().sort_index()
    total_counts = df["total_emoji_count_group"].value_counts().sort_index()
    
    total_posts = len(df)
    unique_percentages = (unique_counts / total_posts * 100).reindex(labels, fill_value=0)
    total_percentages = (total_counts / total_posts * 100).reindex(labels, fill_value=0)
    
    x_pos = np.arange(len(labels))
    
    axes[0].bar(x_pos, unique_percentages, color='yellow', alpha=0.7, label='Unique Emojis (no duplicates)')
    axes[0].set_xlabel('Number of Emojis')
    axes[0].set_ylabel('Percentage')
    axes[0].set_title('Distribution of Emojis - Unique Emojis')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(x_pos, total_percentages, color='brown', alpha=0.7, label='Emojis (total count)')
    axes[1].set_xlabel('Number of Emojis')
    axes[1].set_ylabel('Percentage')
    axes[1].set_title('Distribution of Emojis - Total Count')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, (unique_pct, total_pct) in enumerate(zip(unique_percentages, total_percentages)):
        axes[0].text(i, unique_pct + 1, f'{unique_pct:.1f}%', ha='center', va='bottom', fontsize=8)
        axes[1].text(i, total_pct + 1, f'{total_pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "emoji_distribution_histogram.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Emoji distribution histogram saved to {output_path}")
    plt.show()

def generate_sample_emoji_sentiment_data(output_dir):
    emojis = ['🩸', '📉', '💩', '💀', '😭', '😡', '🤡', '😈', '👻', '🤦', '😩', '😱', '🤯', 
              '🙃', '😅', '🤷', '🤣', '🤔', '🧐', '😴', '🤫', '🤝', '🥳', '🤩', '🐻',
              '💯', '✅', '💰', '📈', '❤️', '💎', '🦍', '🐕', '🚀', '🌙', '💵', '🤑', '🐂']
    
    n_emojis = len(emojis)
    bearish_props = np.linspace(0.95, 0.05, n_emojis)
    bullish_props = 1 - bearish_props
    
    sample_data = {
        'emoji': emojis,
        'bullish_prop': bullish_props,
        'bearish_prop': bearish_props,
        'total_count': np.random.randint(1000, 10000, n_emojis),
        'bullish_count': (bullish_props * np.random.randint(1000, 10000, n_emojis)).astype(int),
        'bearish_count': (bearish_props * np.random.randint(1000, 10000, n_emojis)).astype(int)
    }
    
    df = pd.DataFrame(sample_data)
    df = df.sort_values('bearish_prop', ascending=False)
    
    output_path = Path(output_dir) / "sample_emoji_sentiment_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Sample emoji sentiment data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emoji-sentiment-csv", help="Path to emoji sentiment CSV file")
    parser.add_argument("--emoji-only-csv", help="Path to emoji_only.csv for distribution analysis")
    parser.add_argument("--output-dir", default="replication/plots", help="Output directory")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top emojis to display")
    parser.add_argument("--generate-sample", action="store_true", help="Generate sample data files")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.generate_sample:
        generate_sample_emoji_sentiment_data(args.output_dir)
    
    if args.emoji_sentiment_csv:
        create_emoji_sentiment_bar_chart(args.emoji_sentiment_csv, args.output_dir, args.top_k)
    
    if args.emoji_only_csv:
        create_emoji_distribution_histogram(args.emoji_only_csv, args.output_dir)
