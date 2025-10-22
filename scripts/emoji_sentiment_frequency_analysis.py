import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import re

# Emoji regex pattern (same as in build_dataset.py)
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+"
)

def count_unique_emojis(text):
    """Count the number of unique emojis in text"""
    emojis = emoji_pattern.findall(text)
    return len(set(emojis))

def create_emoji_sentiment_frequency_chart(input_file, output_dir):
    print(f"Loading data from {input_file}...")
    
    # Detect file format and load accordingly
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file, engine='pyarrow', columns=['text', 'sentiment'])
    else:
        df = pd.read_csv(input_file, usecols=['text', 'sentiment'])
    print(f"Loaded {len(df)} total rows")
    
    # Convert text to string and filter for sentiment
    df['text'] = df['text'].astype(str)
    
    # Filter for posts with sentiment labels (Bullish/Bearish)
    df_sentiment = df[df['sentiment'].isin(['Bullish', 'Bearish'])].copy()
    print(f"Posts with sentiment labels: {len(df_sentiment):,}")
    
    # Count unique emojis for each post
    print("Counting unique emojis per post...")
    df_sentiment['unique_emoji_count'] = df_sentiment['text'].apply(count_unique_emojis)
    
    # Filter to posts with at least 1 emoji
    df_emojis = df_sentiment[df_sentiment['unique_emoji_count'] > 0].copy()
    print(f"Posts with emojis and sentiment: {len(df_emojis):,}")
    
    # Create bins for emoji counts (1, 2, 3, ..., 9, 10+)
    df_emojis['emoji_bin'] = df_emojis['unique_emoji_count'].apply(lambda x: min(x, 10))
    
    # Calculate sentiment percentages by emoji count
    sentiment_by_count = df_emojis.groupby('emoji_bin')['sentiment'].value_counts().unstack(fill_value=0)
    
    # Calculate percentages
    sentiment_pct = sentiment_by_count.div(sentiment_by_count.sum(axis=1), axis=0) * 100
    
    # Calculate frequency (total posts) by emoji count
    frequency_by_count = df_emojis['emoji_bin'].value_counts().sort_index()
    
    # Ensure we have data for all bins 1-10
    bins = list(range(1, 11))
    for bin_val in bins:
        if bin_val not in sentiment_pct.index:
            sentiment_pct.loc[bin_val] = {'Bullish': 0, 'Bearish': 0}
        if bin_val not in frequency_by_count.index:
            frequency_by_count.loc[bin_val] = 0
    
    # Sort by bin value
    sentiment_pct = sentiment_pct.sort_index()
    frequency_by_count = frequency_by_count.sort_index()
    
    # Create the combined chart
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Create x-axis labels
    x_labels = [str(i) if i <= 9 else '10+' for i in bins]
    x_pos = np.arange(len(bins))
    
    # Plot sentiment bars
    bullish_pct = sentiment_pct['Bullish'].values if 'Bullish' in sentiment_pct.columns else [0] * len(bins)
    bearish_pct = sentiment_pct['Bearish'].values if 'Bearish' in sentiment_pct.columns else [0] * len(bins)
    
    width = 0.35
    bars1 = ax1.bar(x_pos - width/2, bullish_pct, width, label='Bullish', color='green', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, bearish_pct, width, label='Bearish', color='red', alpha=0.7)
    
    # Set up first y-axis (sentiment percentage)
    ax1.set_xlabel('Emoji Count', fontsize=12)
    ax1.set_ylabel('Sentiment Percentage (%)', fontsize=12)
    ax1.set_title('Emoji Count versus Sentiment and Frequency', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim(0, 80)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Create second y-axis for frequency
    ax2 = ax1.twinx()
    
    # Plot frequency line
    frequency_values = frequency_by_count.values
    line = ax2.plot(x_pos, frequency_values, 'b-o', linewidth=2, markersize=6, 
                   label='Frequency', color='lightblue', markerfacecolor='blue')
    ax2.set_ylabel('Frequency (Number of Posts)', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Add frequency values as labels above the line
    for i, (x, y) in enumerate(zip(x_pos, frequency_values)):
        ax2.annotate(f'{int(y):,}', (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9, color='blue')
    
    # Add legend for the line
    ax2.legend(loc='upper left')
    
    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.annotate(f'{value:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1, bullish_pct)
    add_value_labels(bars2, bearish_pct)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / "emoji_sentiment_frequency_chart.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {plot_path}")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("EMOJI COUNT vs SENTIMENT AND FREQUENCY SUMMARY")
    print("="*80)
    print(f"{'Count':<8} {'Bullish %':<12} {'Bearish %':<12} {'Frequency':<12}")
    print("-" * 50)
    
    for i, bin_val in enumerate(bins):
        bullish_val = bullish_pct[i]
        bearish_val = bearish_pct[i]
        freq_val = int(frequency_values[i])
        label = str(bin_val) if bin_val <= 9 else '10+'
        print(f"{label:<8} {bullish_val:<12.1f} {bearish_val:<12.1f} {freq_val:<12,}")
    
    # Save data to CSV
    results_df = pd.DataFrame({
        'Emoji_Count': x_labels,
        'Bullish_Percentage': bullish_pct,
        'Bearish_Percentage': bearish_pct,
        'Frequency': frequency_values
    })
    
    csv_path = output_path / "emoji_sentiment_frequency_data.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nData saved to {csv_path}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Create emoji count vs sentiment and frequency chart.")
    parser.add_argument("--input", required=True, help="Path to the input Parquet file (e.g., output.parquet)")
    parser.add_argument("--output-dir", required=True, help="Directory to save the chart and data")
    args = parser.parse_args()
    
    create_emoji_sentiment_frequency_chart(args.input, args.output_dir)

if __name__ == "__main__":
    main()
