import pandas as pd
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

def extract_emojis(text):
    return emoji_pattern.findall(text)

def calculate_length_statistics(input_file, output_dir):
    print(f"Loading data from {input_file}...")
    
    # Detect file format and load accordingly
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file, engine='pyarrow', columns=['text'])
    else:
        df = pd.read_csv(input_file, usecols=['text'])
    print(f"Loaded {len(df)} total rows")
    
    # Convert text to string and clean
    df['text'] = df['text'].astype(str)
    
    # Calculate text lengths (character count)
    print("Calculating text lengths...")
    df['text_length'] = df['text'].str.len()
    
    # Calculate emoji lengths (character count of emojis only)
    print("Extracting emojis and calculating emoji lengths...")
    df['emojis'] = df['text'].apply(extract_emojis)
    df['emoji_text'] = df['emojis'].apply(lambda x: ''.join(x))
    df['emoji_length'] = df['emoji_text'].str.len()
    
    # Filter out rows where both text and emoji lengths are 0 (empty posts)
    df = df[(df['text_length'] > 0) | (df['emoji_length'] > 0)]
    print(f"After filtering empty posts: {len(df)} rows")
    
    # Calculate percentiles
    percentiles = [5, 25, 50, 75, 90, 95, 99]
    
    text_percentiles = np.percentile(df['text_length'], percentiles)
    emoji_percentiles = np.percentile(df['emoji_length'], percentiles)
    
    # Create the results table
    results_data = {
        'Percentile': ['5th', '25th', '50th', '75th', '90th', '95th', '99th'],
        'Text Length': [round(val, 2) for val in text_percentiles],
        'Emoji Length': [round(val, 2) for val in emoji_percentiles]
    }
    
    results_df = pd.DataFrame(results_data)
    
    # Print the table
    print("\n" + "="*80)
    print("PERCENTILE DISTRIBUTION OF TEXT AND EMOJI LENGTHS")
    print("="*80)
    print(results_df.to_string(index=False, float_format='%.2f'))
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path / "text_emoji_length_percentiles.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Print additional statistics
    print(f"\nADDITIONAL STATISTICS:")
    print(f"Total posts analyzed: {len(df):,}")
    print(f"Posts with text only: {len(df[df['emoji_length'] == 0]):,}")
    print(f"Posts with emojis only: {len(df[df['text_length'] == 0]):,}")
    print(f"Posts with both text and emojis: {len(df[(df['text_length'] > 0) & (df['emoji_length'] > 0)]):,}")
    
    print(f"\nText length statistics:")
    print(f"  Mean: {df['text_length'].mean():.2f} characters")
    print(f"  Median: {df['text_length'].median():.2f} characters")
    print(f"  Max: {df['text_length'].max()} characters")
    
    print(f"\nEmoji length statistics:")
    print(f"  Mean: {df['emoji_length'].mean():.2f} characters")
    print(f"  Median: {df['emoji_length'].median():.2f} characters")
    print(f"  Max: {df['emoji_length'].max()} characters")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(results_df)
    latex_path = output_path / "text_emoji_length_percentiles.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {latex_path}")
    
    return results_df

def generate_latex_table(df):
    text_vals = df['Text Length'].tolist()
    emoji_vals = df['Emoji Length'].tolist()
    
    latex = """\\begin{{table}}[ht]
\\centering
\\caption{{\\label{{table:tokenlengthsPercentile}}Percentile Distribution of Text and Emoji Lengths in Posts. This table displays the character length distribution for text-only and emoji-only content across various percentiles, demonstrating the comparative brevity of emoji usage.}}
\\begin{{tabular}}{{lccccccc}}
\\toprule
\\textbf{{Percentile}} & 5$^\\mathrm{{th}}$ & 25$^\\mathrm{{th}}$ & 50$^\\mathrm{{th}}$ & 75$^\\mathrm{{th}}$ & 90$^\\mathrm{{th}}$ & 95$^\\mathrm{{th}}$ & 99$^\\mathrm{{th}}$ \\\\ \\midrule
\\textbf{{Text Length}} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ 
\\textbf{{Emoji Length}} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\ \\bottomrule
\\end{{tabular}}
\\end{{table}}""".format(
        *text_vals,
        *emoji_vals
    )
    return latex

def main():
    parser = argparse.ArgumentParser(description="Calculate percentile statistics for text and emoji lengths.")
    parser.add_argument("--input", required=True, help="Path to the input Parquet file (e.g., output.parquet)")
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    args = parser.parse_args()
    
    calculate_length_statistics(args.input, args.output_dir)

if __name__ == "__main__":
    main()
