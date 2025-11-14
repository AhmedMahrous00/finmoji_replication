import pandas as pd
import numpy as np
import re
from collections import Counter
from tqdm import tqdm
from pathlib import Path

emoji_pattern = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF\u2700-\u27BF]"
)

def remove_emojis(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return emoji_pattern.sub("", str(text))

def extract_words(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    text_no_emojis = remove_emojis(text)
    words = re.findall(r'\b\w+\b', text_no_emojis.lower())
    return [w for w in words if len(w) > 0]

def compute_entropy(frequencies):
    if not frequencies:
        return 0.0
    probs = np.array(frequencies, dtype=np.float64) / sum(frequencies)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def get_top_90_percent_by_frequency(counts_dict, desc="Finding top 90%"):
    if not counts_dict:
        return {}
    
    print(f"{desc}: Sorting {len(counts_dict):,} items...")
    sorted_items = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
    total_freq = sum(counts_dict.values())
    target_freq = 0.90 * total_freq
    
    print(f"{desc}: Searching for top 90% (target: {target_freq:,.0f} of {total_freq:,.0f} occurrences)...")
    cumulative = 0
    top_90_items = {}
    for item, freq in tqdm(sorted_items, desc=desc, unit="items"):
        top_90_items[item] = freq
        cumulative += freq
        if cumulative >= target_freq:
            break
    
    return top_90_items

def main():
    input_file = "emojitweets.csv"
    output_file = "entropy_analysis_results.csv"
    
    chunk_size = 100000
    
    print(f"Counting total rows in {input_file}...")
    total_rows = 0
    pbar = tqdm(desc="Counting rows", unit=" rows")
    try:
        for chunk in pd.read_csv(input_file, chunksize=chunk_size, usecols=['post_id']):
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            pbar.update(chunk_rows)
    finally:
        pbar.close()
    
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    print(f"Found {total_rows:,} total rows ({num_chunks} chunks of size {chunk_size:,})")
    print(f"Estimated processing time will be shown as chunks complete.\n")
    print(f"Starting processing...\n")
    
    emoji_counts = Counter()
    word_counts = Counter()
    
    chunk_iter = pd.read_csv(input_file, chunksize=chunk_size)
    for chunk_idx, chunk in enumerate(tqdm(chunk_iter, total=num_chunks, desc="Processing chunks", unit="chunk"), 1):
        if 'emojis' in chunk.columns:
            emoji_series = chunk['emojis'].dropna().astype(str)
            if len(emoji_series) > 0:
                all_emojis = []
                for s in emoji_series:
                    emojis = str(s).strip().split()
                    if emojis:
                        all_emojis.extend(emojis)
                if all_emojis:
                    emoji_counts.update(all_emojis)
        
        if 'text' in chunk.columns:
            text_series = chunk['text'].dropna().astype(str)
            if len(text_series) > 0:
                all_words = []
                for s in text_series:
                    words = extract_words(s)
                    if words:
                        all_words.extend(words)
                if all_words:
                    word_counts.update(all_words)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
    print(f"Processed {total_rows:,} rows")
    print(f"Total unique emojis: {len(emoji_counts):,}")
    print(f"Total unique words: {len(word_counts):,}")
    print(f"Total emoji occurrences: {sum(emoji_counts.values()):,}")
    print(f"Total word occurrences: {sum(word_counts.values()):,}")
    
    print("\n" + "="*60)
    print("Finding top 90% by cumulative frequency...")
    print("="*60)
    top_90_emojis = get_top_90_percent_by_frequency(emoji_counts, desc="Emojis")
    top_90_words = get_top_90_percent_by_frequency(word_counts, desc="Words")
    
    print(f"\nTop 90% emojis: {len(top_90_emojis):,} unique emojis")
    print(f"Top 90% words: {len(top_90_words):,} unique words")
    
    print("\n" + "="*60)
    print("Computing entropy...")
    print("="*60)
    print("Computing emoji entropy...")
    emoji_entropy = compute_entropy(list(top_90_emojis.values()))
    print("Computing word entropy...")
    word_entropy = compute_entropy(list(top_90_words.values()))
    
    print("\nComputing average word length...")
    avg_word_length = np.mean([len(word) for word in tqdm(top_90_words.keys(), desc="Calculating lengths", unit="words")]) if top_90_words else 0.0
    
    results = {
        'metric': [
            'emoji_entropy_top90',
            'word_entropy_top90',
            'avg_word_length_top90',
            'num_unique_emojis_top90',
            'num_unique_words_top90',
            'total_emoji_occurrences',
            'total_word_occurrences',
            'total_unique_emojis',
            'total_unique_words'
        ],
        'value': [
            emoji_entropy,
            word_entropy,
            avg_word_length,
            len(top_90_emojis),
            len(top_90_words),
            sum(emoji_counts.values()),
            sum(word_counts.values()),
            len(emoji_counts),
            len(word_counts)
        ]
    }
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Emoji Entropy (top 90%): {emoji_entropy:.4f} bits")
    print(f"Word Entropy (top 90%): {word_entropy:.4f} bits")
    print(f"Average Word Length (top 90%): {avg_word_length:.2f} characters")
    print(f"\nTop 90% covers {len(top_90_emojis):,} unique emojis")
    print(f"Top 90% covers {len(top_90_words):,} unique words")
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

