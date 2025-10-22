import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Emoji pattern for detection
emoji_pattern = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF\u2700-\u27BF]"
)

def has_emoji(text):
    """Check if text contains at least one emoji"""
    if pd.isna(text):
        return False
    return bool(emoji_pattern.search(str(text)))

def create_data_pipeline_visualization(input_file, output_dir):
    print(f"Loading data from {input_file}...")
    
    # Detect file format and load accordingly
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file, engine='pyarrow', columns=['text', 'sentiment'])
    else:
        df = pd.read_csv(input_file, usecols=['text', 'sentiment'])
    print(f"Loaded {len(df)} total rows")
    
    # Convert text to string and check for emojis
    print("Analyzing emoji presence...")
    df['text'] = df['text'].astype(str)
    df['has_emoji'] = df['text'].apply(has_emoji)
    
    # Stage 1: All Posts
    print("Stage 1: Analyzing all posts...")
    all_posts = len(df)
    sentiment_counts_all = df['sentiment'].value_counts()
    
    # Stage 2: Posts with Emojis
    print("Stage 2: Analyzing posts with emojis...")
    emoji_posts = df[df['has_emoji']].copy()
    posts_with_emojis = len(emoji_posts)
    sentiment_counts_emoji = emoji_posts['sentiment'].value_counts()
    
    # Stage 3: Labelled Posts with Emojis (Bullish/Bearish only)
    print("Stage 3: Analyzing labelled posts with emojis...")
    labeled_emoji_posts = emoji_posts[emoji_posts['sentiment'].isin(['Bullish', 'Bearish'])].copy()
    labeled_posts_with_emojis = len(labeled_emoji_posts)
    sentiment_counts_labeled = labeled_emoji_posts['sentiment'].value_counts()
    
    # Stage 4: Balanced (assuming equal split for visualization)
    print("Stage 4: Analyzing balanced dataset...")
    # For balanced, we'll use the minimum count between Bullish and Bearish
    bullish_count = sentiment_counts_labeled.get('Bullish', 0)
    bearish_count = sentiment_counts_labeled.get('Bearish', 0)
    balanced_count = min(bullish_count, bearish_count) * 2  # Both classes
    
    # Create the visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Define colors to match the image
    colors = {
        'Bullish': '#90EE90',  # Light green
        'Bearish': '#FFB6C1',  # Light pink  
        'Other': '#D3D3D3',    # Light gray
        'Unlabeled': '#D3D3D3' # Light gray for unlabeled posts
    }
    
    # Stage 1: All Posts
    ax1 = axes[0]
    all_sizes = []
    all_labels = []
    all_colors = []
    
    # Calculate unlabeled posts (posts that are neither Bullish nor Bearish)
    bullish_count_all = sentiment_counts_all.get('Bullish', 0)
    bearish_count_all = sentiment_counts_all.get('Bearish', 0)
    other_count_all = sentiment_counts_all.get('Other', 0)
    unlabeled_count = all_posts - bullish_count_all - bearish_count_all - other_count_all
    
    # Add Bullish, Bearish, Other, and Unlabeled
    for sentiment, count in [('Bullish', bullish_count_all), ('Bearish', bearish_count_all), ('Other', other_count_all), ('Unlabeled', unlabeled_count)]:
        if count > 0:
            all_sizes.append(count)
            all_labels.append(f'{sentiment}\n{count/all_posts*100:.0f}%')
            all_colors.append(colors[sentiment])
    
    ax1.pie(all_sizes, labels=all_labels, colors=all_colors, autopct='', startangle=90)
    ax1.set_title('All Posts', fontsize=14, fontweight='bold')
    ax1.text(0, -1.3, f'Total Posts: {all_posts:,}', ha='center', fontsize=12, fontweight='bold')
    
    # Stage 2: Posts with Emojis
    ax2 = axes[1]
    emoji_sizes = []
    emoji_labels = []
    emoji_colors = []
    
    # Calculate unlabeled posts with emojis
    bullish_count_emoji = sentiment_counts_emoji.get('Bullish', 0)
    bearish_count_emoji = sentiment_counts_emoji.get('Bearish', 0)
    other_count_emoji = sentiment_counts_emoji.get('Other', 0)
    unlabeled_count_emoji = posts_with_emojis - bullish_count_emoji - bearish_count_emoji - other_count_emoji
    
    # Add Bullish, Bearish, Other, and Unlabeled
    for sentiment, count in [('Bullish', bullish_count_emoji), ('Bearish', bearish_count_emoji), ('Other', other_count_emoji), ('Unlabeled', unlabeled_count_emoji)]:
        if count > 0:
            emoji_sizes.append(count)
            emoji_labels.append(f'{sentiment}\n{count/posts_with_emojis*100:.0f}%')
            emoji_colors.append(colors[sentiment])
    
    ax2.pie(emoji_sizes, labels=emoji_labels, colors=emoji_colors, autopct='', startangle=90)
    ax2.set_title('Posts with Emojis', fontsize=14, fontweight='bold')
    ax2.text(0, -1.3, f'Total Posts: {posts_with_emojis:,}', ha='center', fontsize=12, fontweight='bold')
    
    
    # Stage 3: Labelled Posts with Emojis
    ax3 = axes[2]
    labeled_sizes = []
    labeled_labels = []
    labeled_colors = []
    
    for sentiment in ['Bullish', 'Bearish']:
        count = sentiment_counts_labeled.get(sentiment, 0)
        if count > 0:
            labeled_sizes.append(count)
            labeled_labels.append(f'{sentiment}\n{count/labeled_posts_with_emojis*100:.0f}%')
            labeled_colors.append(colors[sentiment])
    
    ax3.pie(labeled_sizes, labels=labeled_labels, colors=labeled_colors, autopct='', startangle=90)
    ax3.set_title('Labelled Posts with Emojis', fontsize=14, fontweight='bold')
    ax3.text(0, -1.3, f'Total Posts: {labeled_posts_with_emojis:,}', ha='center', fontsize=12, fontweight='bold')
    
    # Stage 4: Balanced
    ax4 = axes[3]
    balanced_sizes = [balanced_count // 2, balanced_count // 2]
    balanced_labels = [f'Bullish\n50%', f'Bearish\n50%']
    balanced_colors = [colors['Bullish'], colors['Bearish']]
    
    ax4.pie(balanced_sizes, labels=balanced_labels, colors=balanced_colors, autopct='', startangle=90)
    ax4.set_title('Balanced', fontsize=14, fontweight='bold')
    ax4.text(0, -1.3, f'Total Posts: {balanced_count:,}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / "data_pipeline_pie_charts.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Pipeline visualization saved to {plot_path}")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("DATA PIPELINE SUMMARY")
    print("="*80)
    print(f"Stage 1 - All Posts: {all_posts:,}")
    print(f"  Bullish: {bullish_count_all:,} ({bullish_count_all/all_posts*100:.1f}%)")
    print(f"  Bearish: {bearish_count_all:,} ({bearish_count_all/all_posts*100:.1f}%)")
    print(f"  Other: {other_count_all:,} ({other_count_all/all_posts*100:.1f}%)")
    print(f"  Unlabeled: {unlabeled_count:,} ({unlabeled_count/all_posts*100:.1f}%)")
    
    print(f"\nStage 2 - Posts with Emojis: {posts_with_emojis:,}")
    print(f"  Bullish: {bullish_count_emoji:,} ({bullish_count_emoji/posts_with_emojis*100:.1f}%)")
    print(f"  Bearish: {bearish_count_emoji:,} ({bearish_count_emoji/posts_with_emojis*100:.1f}%)")
    print(f"  Other: {other_count_emoji:,} ({other_count_emoji/posts_with_emojis*100:.1f}%)")
    print(f"  Unlabeled: {unlabeled_count_emoji:,} ({unlabeled_count_emoji/posts_with_emojis*100:.1f}%)")
    
    print(f"\nStage 3 - Labelled Posts with Emojis: {labeled_posts_with_emojis:,}")
    print(f"  Bullish: {sentiment_counts_labeled.get('Bullish', 0):,} ({sentiment_counts_labeled.get('Bullish', 0)/labeled_posts_with_emojis*100:.1f}%)")
    print(f"  Bearish: {sentiment_counts_labeled.get('Bearish', 0):,} ({sentiment_counts_labeled.get('Bearish', 0)/labeled_posts_with_emojis*100:.1f}%)")
    
    print(f"\nStage 4 - Balanced: {balanced_count:,}")
    print(f"  Bullish: {balanced_count//2:,} (50.0%)")
    print(f"  Bearish: {balanced_count//2:,} (50.0%)")
    
    # Save detailed statistics
    stats_data = {
        'Stage': ['All Posts', 'Posts with Emojis', 'Labelled Posts with Emojis', 'Balanced'],
        'Total_Posts': [all_posts, posts_with_emojis, labeled_posts_with_emojis, balanced_count],
        'Bullish_Count': [bullish_count_all, bullish_count_emoji, sentiment_counts_labeled.get('Bullish', 0), balanced_count//2],
        'Bearish_Count': [bearish_count_all, bearish_count_emoji, sentiment_counts_labeled.get('Bearish', 0), balanced_count//2],
        'Other_Count': [other_count_all, other_count_emoji, 0, 0],
        'Unlabeled_Count': [unlabeled_count, unlabeled_count_emoji, 0, 0]
    }
    
    stats_df = pd.DataFrame(stats_data)
    stats_path = output_path / "pipeline_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Pipeline statistics saved to {stats_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create data pipeline visualization with pie charts")
    parser.add_argument("--input", required=True, help="Path to input parquet file")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    args = parser.parse_args()
    
    create_data_pipeline_visualization(args.input, args.output_dir)
