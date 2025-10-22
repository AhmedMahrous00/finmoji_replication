import pandas as pd
import numpy as np
from pathlib import Path

def calculate_percentile_statistics(input_csv_path, output_dir):
    print(f"Loading {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    print("Calculating text and emoji lengths...")
    df['text_length'] = df['text'].str.len()
    
    # Load emoji_only data for emoji length calculation
    emoji_df = pd.read_csv(input_csv_path.replace('text_emoji.csv', 'emoji_only.csv'))
    df['emoji_length'] = emoji_df['text'].str.len()

    print("Calculating percentiles...")
    percentiles = [5, 25, 50, 75, 90, 95, 99]
    text_stats = np.percentile(df['text_length'], percentiles)
    emoji_stats = np.percentile(df['emoji_length'], percentiles)

    print("\nPercentile Statistics:")
    print("=" * 60)
    print(f"{'Percentile':<12} {'5th':<8} {'25th':<8} {'50th':<8} {'75th':<8} {'90th':<8} {'95th':<8} {'99th':<8}")
    print("-" * 60)
    print(f"{'Text Length':<12} {text_stats[0]:<8.2f} {text_stats[1]:<8.2f} {text_stats[2]:<8.2f} {text_stats[3]:<8.2f} {text_stats[4]:<8.2f} {text_stats[5]:<8.2f} {text_stats[6]:<8.2f}")
    print(f"{'Emoji Length':<12} {emoji_stats[0]:<8.2f} {emoji_stats[1]:<8.2f} {emoji_stats[2]:<8.2f} {emoji_stats[3]:<8.2f} {emoji_stats[4]:<8.2f} {emoji_stats[5]:<8.2f} {emoji_stats[6]:<8.2f}")

    output_path = Path(output_dir) / "percentile_statistics.csv"
    results_df = pd.DataFrame({
        'Metric': ['Text Length', 'Emoji Length'],
        '5th': [text_stats[0], emoji_stats[0]],
        '25th': [text_stats[1], emoji_stats[1]],
        '50th': [text_stats[2], emoji_stats[2]],
        '75th': [text_stats[3], emoji_stats[3]],
        '90th': [text_stats[4], emoji_stats[4]],
        '95th': [text_stats[5], emoji_stats[5]],
        '99th': [text_stats[6], emoji_stats[6]]
    })
    
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to dataset CSV file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    calculate_percentile_statistics(args.input, args.output_dir)
