import pandas as pd
import numpy as np
import re
from pathlib import Path

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

def quick_emoji_analysis(input_file, output_dir):
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
    
    # Get sentiment distribution
    print("\n" + "="*80)
    print("EMOJI DISTRIBUTION BY SENTIMENT CLASS")
    print("="*80)
    
    # Overall sentiment distribution
    sentiment_counts = df['sentiment'].value_counts(dropna=False)
    print(f"\nTotal posts by sentiment:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment}: {count:,} ({percentage:.2f}%)")
    
    # Emoji distribution by sentiment
    print(f"\nPosts with at least one emoji by sentiment:")
    emoji_by_sentiment = df.groupby('sentiment')['has_emoji'].agg(['count', 'sum']).reset_index()
    emoji_by_sentiment.columns = ['sentiment', 'total_posts', 'posts_with_emoji']
    emoji_by_sentiment['posts_without_emoji'] = emoji_by_sentiment['total_posts'] - emoji_by_sentiment['posts_with_emoji']
    emoji_by_sentiment['emoji_proportion'] = (emoji_by_sentiment['posts_with_emoji'] / emoji_by_sentiment['total_posts']) * 100
    
    for _, row in emoji_by_sentiment.iterrows():
        print(f"  {row['sentiment']}:")
        print(f"    With emoji: {row['posts_with_emoji']:,} ({row['emoji_proportion']:.2f}%)")
        print(f"    Without emoji: {row['posts_without_emoji']:,} ({100-row['emoji_proportion']:.2f}%)")
    
    # Overall emoji statistics
    total_with_emoji = df['has_emoji'].sum()
    total_without_emoji = len(df) - total_with_emoji
    overall_emoji_proportion = (total_with_emoji / len(df)) * 100
    
    print(f"\nOverall emoji statistics:")
    print(f"  Total posts: {len(df):,}")
    print(f"  Posts with emoji: {total_with_emoji:,} ({overall_emoji_proportion:.2f}%)")
    print(f"  Posts without emoji: {total_without_emoji:,} ({100-overall_emoji_proportion:.2f}%)")
    
    # Detailed breakdown for labeled posts (Bullish/Bearish)
    labeled_df = df[df['sentiment'].isin(['Bullish', 'Bearish'])].copy()
    if len(labeled_df) > 0:
        print(f"\n" + "="*80)
        print("LABELED POSTS (BULLISH/BEARISH) EMOJI DISTRIBUTION")
        print("="*80)
        
        labeled_emoji_by_sentiment = labeled_df.groupby('sentiment')['has_emoji'].agg(['count', 'sum']).reset_index()
        labeled_emoji_by_sentiment.columns = ['sentiment', 'total_posts', 'posts_with_emoji']
        labeled_emoji_by_sentiment['posts_without_emoji'] = labeled_emoji_by_sentiment['total_posts'] - labeled_emoji_by_sentiment['posts_with_emoji']
        labeled_emoji_by_sentiment['emoji_proportion'] = (labeled_emoji_by_sentiment['posts_with_emoji'] / labeled_emoji_by_sentiment['total_posts']) * 100
        
        print(f"\nLabeled posts with emoji by sentiment:")
        for _, row in labeled_emoji_by_sentiment.iterrows():
            print(f"  {row['sentiment']}:")
            print(f"    With emoji: {row['posts_with_emoji']:,} ({row['emoji_proportion']:.2f}%)")
            print(f"    Without emoji: {row['posts_without_emoji']:,} ({100-row['emoji_proportion']:.2f}%)")
        
        # Cross-tabulation
        print(f"\nCross-tabulation (sentiment vs emoji presence):")
        crosstab = pd.crosstab(labeled_df['sentiment'], labeled_df['has_emoji'], margins=True)
        crosstab_percent = pd.crosstab(labeled_df['sentiment'], labeled_df['has_emoji'], normalize='index') * 100
        
        print(f"\nCounts:")
        print(crosstab)
        
        print(f"\nPercentages (by row):")
        print(crosstab_percent.round(2))
    
    # Save results to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    emoji_by_sentiment.to_csv(output_path / "emoji_distribution_by_sentiment.csv", index=False)
    
    if len(labeled_df) > 0:
        labeled_emoji_by_sentiment.to_csv(output_path / "labeled_emoji_distribution.csv", index=False)
        crosstab.to_csv(output_path / "emoji_sentiment_crosstab_counts.csv")
        crosstab_percent.to_csv(output_path / "emoji_sentiment_crosstab_percentages.csv")
    
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Analyzed {len(df):,} posts")
    print(f"✓ {total_with_emoji:,} posts contain at least one emoji ({overall_emoji_proportion:.2f}%)")
    print(f"✓ Results saved to {output_path}/")
    
    return {
        'total_posts': len(df),
        'posts_with_emoji': total_with_emoji,
        'emoji_proportion': overall_emoji_proportion,
        'sentiment_distribution': sentiment_counts.to_dict(),
        'emoji_by_sentiment': emoji_by_sentiment
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick emoji distribution analysis by sentiment class")
    parser.add_argument("--input", required=True, help="Path to input parquet file")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    args = parser.parse_args()
    
    quick_emoji_analysis(args.input, args.output_dir)
