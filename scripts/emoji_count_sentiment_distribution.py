import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import argparse
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set publication-style plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'figure.dpi': 300
})

class EmojiCountSentimentAnalyzer:
    def __init__(self, dataset_path, output_dir):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Emoji regex pattern (comprehensive Unicode ranges)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002600-\U000026FF"  # miscellaneous symbols
            "\U00002700-\U000027BF"  # dingbats
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
            "]+"
        )
        
        # Define bins: [0,1,2,3,4,5,6,7,8,9,≥10]
        self.bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]
        self.bin_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '≥10']
        
        self.results = None

    def extract_unique_emojis(self, text):
        """Extract unique emojis from text"""
        if pd.isna(text) or text == '':
            return set()
        
        # Find all emoji sequences
        emoji_matches = self.emoji_pattern.findall(str(text))
        
        # Flatten and get unique emojis
        unique_emojis = set()
        for match in emoji_matches:
            # Each match might contain multiple emojis, so we need to split them
            for char in match:
                if self.emoji_pattern.match(char):
                    unique_emojis.add(char)
        
        return unique_emojis

    def count_unique_emojis(self, text):
        """Count the number of unique emojis in text"""
        unique_emojis = self.extract_unique_emojis(text)
        return len(unique_emojis)

    def load_and_process_data(self):
        """Load dataset and process emoji counts"""
        print(f"Loading data from {self.dataset_path}...")
        
        # Load the dataset
        df = pd.read_csv(self.dataset_path)
        print(f"Loaded {len(df):,} posts")
        
        # Filter out posts without labels
        df = df.dropna(subset=['label'])
        print(f"After filtering for labels: {len(df):,} posts")
        
        # Determine text column
        if 'text' in df.columns:
            text_col = 'text'
        elif 'emojis' in df.columns:
            text_col = 'emojis'
        else:
            raise ValueError("No 'text' or 'emojis' column found in dataset")
        
        print(f"Using column: {text_col}")
        
        # Clean text data
        df[text_col] = df[text_col].fillna('').astype(str)
        
        # Count unique emojis for each post
        print("Counting unique emojis per post...")
        df['unique_emoji_count'] = df[text_col].apply(self.count_unique_emojis)
        
        # Filter out posts with no emojis if needed (optional)
        # df = df[df['unique_emoji_count'] > 0]
        
        print(f"Emoji count statistics:")
        print(f"  Mean: {df['unique_emoji_count'].mean():.2f}")
        print(f"  Median: {df['unique_emoji_count'].median():.2f}")
        print(f"  Max: {df['unique_emoji_count'].max()}")
        print(f"  Posts with ≥5 emojis: {len(df[df['unique_emoji_count'] >= 5]):,}")
        print(f"  Posts with ≥10 emojis: {len(df[df['unique_emoji_count'] >= 10]):,}")
        
        return df

    def analyze_emoji_count_distribution(self, df):
        """Analyze the distribution of bullish vs bearish posts by emoji count"""
        print("Analyzing emoji count distribution...")
        
        # Create bins for emoji counts
        df['emoji_bin'] = pd.cut(df['unique_emoji_count'], 
                                bins=self.bins, 
                                labels=self.bin_labels, 
                                include_lowest=True, 
                                right=False)
        
        # Group by emoji bin and sentiment
        distribution = df.groupby(['emoji_bin', 'label']).size().unstack(fill_value=0)
        
        # Ensure we have both bullish and bearish columns
        if 'bullish' not in distribution.columns:
            distribution['bullish'] = 0
        if 'bearish' not in distribution.columns:
            distribution['bearish'] = 0
        
        # Calculate totals and ratios
        distribution['total_count'] = distribution['bullish'] + distribution['bearish']
        distribution['bullish_ratio'] = distribution['bullish'] / distribution['total_count']
        
        # Create bin range labels
        bin_ranges = []
        for i, label in enumerate(self.bin_labels):
            if label == '≥10':
                bin_ranges.append('≥10')
            else:
                bin_ranges.append(f'{label}')
        
        # Prepare results
        results_data = []
        for i, (bin_label, row) in enumerate(distribution.iterrows()):
            results_data.append({
                'bin_label': bin_label,
                'emoji_count_min': self.bins[i] if i < len(self.bins) - 1 else 10,
                'emoji_count_max': self.bins[i + 1] - 1 if i < len(self.bins) - 2 else '∞',
                'bullish_count': int(row['bullish']),
                'bearish_count': int(row['bearish']),
                'total_count': int(row['total_count']),
                'bullish_ratio': round(row['bullish_ratio'], 4)
            })
        
        self.results = pd.DataFrame(results_data)
        
        print("\nEmoji Count Distribution Results:")
        print("=" * 80)
        print(self.results.to_string(index=False, float_format='%.4f'))
        
        return self.results

    def create_visualization(self, results_df):
        """Create the stacked bar chart with frequency line overlay"""
        print("Creating visualization...")
        
        # Set up the plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Create the stacked bar chart
        x_pos = np.arange(len(results_df))
        width = 0.8
        
        # Plot stacked bars
        bullish_bars = ax1.bar(x_pos, results_df['bullish_count'], width, 
                              label='Bullish', color='#2E8B57', alpha=0.8)
        bearish_bars = ax1.bar(x_pos, results_df['bearish_count'], width, 
                              bottom=results_df['bullish_count'], 
                              label='Bearish', color='#DC143C', alpha=0.8)
        
        # Set up second y-axis for bullish ratio
        ax2 = ax1.twinx()
        
        # Plot bullish ratio line
        line = ax2.plot(x_pos, results_df['bullish_ratio'], 
                       'o-', color='black', linewidth=2.5, markersize=6,
                       label='Bullish Ratio', markerfacecolor='white', 
                       markeredgecolor='black', markeredgewidth=1.5)
        
        # Customize axes
        ax1.set_xlabel('Number of Unique Emojis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Posts', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Bullish Ratio', fontsize=14, fontweight='bold')
        
        # Set x-axis labels
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(results_df['bin_label'])
        
        # Set y-axis limits
        ax1.set_ylim(0, results_df['total_count'].max() * 1.1)
        ax2.set_ylim(0, 1.0)
        
        # Add grid
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Add annotations for ≥5 and ≥10 emoji posts
        # Find the indices for ≥5 and ≥10 bins
        ge5_idx = results_df[results_df['bin_label'] == '5'].index[0] if '5' in results_df['bin_label'].values else None
        ge10_idx = results_df[results_df['bin_label'] == '≥10'].index[0] if '≥10' in results_df['bin_label'].values else None
        
        # Add vertical lines and annotations
        if ge5_idx is not None:
            ax1.axvline(x=ge5_idx, color='gray', linestyle=':', alpha=0.7, linewidth=2)
            ax1.annotate('≥5 emoji posts', xy=(ge5_idx, ax1.get_ylim()[1] * 0.9), 
                        xytext=(ge5_idx + 1, ax1.get_ylim()[1] * 0.95),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                        fontsize=10, ha='left', color='gray')
        
        if ge10_idx is not None:
            ax1.axvline(x=ge10_idx, color='gray', linestyle=':', alpha=0.7, linewidth=2)
            ax1.annotate('≥10 emoji posts', xy=(ge10_idx, ax1.get_ylim()[1] * 0.8), 
                        xytext=(ge10_idx - 1, ax1.get_ylim()[1] * 0.85),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                        fontsize=10, ha='right', color='gray')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12)
        
        # Set title
        plt.title('Emoji Count vs Sentiment Distribution\n("Pictogram Rant" Effect Analysis)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / "emoji_count_sentiment_distribution.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {plot_path}")
        
        # Show the plot
        plt.show()
        
        return plot_path

    def save_results(self, results_df):
        """Save results to CSV"""
        csv_path = self.output_dir / "emoji_count_sentiment_distribution.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        return csv_path

    def generate_summary_statistics(self, df, results_df):
        """Generate summary statistics"""
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        total_posts = len(df)
        bullish_posts = len(df[df['label'] == 'bullish'])
        bearish_posts = len(df[df['label'] == 'bearish'])
        
        print(f"Total posts analyzed: {total_posts:,}")
        print(f"Bullish posts: {bullish_posts:,} ({bullish_posts/total_posts*100:.1f}%)")
        print(f"Bearish posts: {bearish_posts:,} ({bearish_posts/total_posts*100:.1f}%)")
        
        # Posts with ≥5 emojis
        ge5_posts = df[df['unique_emoji_count'] >= 5]
        ge5_bullish = len(ge5_posts[ge5_posts['label'] == 'bullish'])
        ge5_bearish = len(ge5_posts[ge5_posts['label'] == 'bearish'])
        
        print(f"\nPosts with ≥5 emojis: {len(ge5_posts):,} ({len(ge5_posts)/total_posts*100:.1f}%)")
        print(f"  Bullish: {ge5_bullish:,} ({ge5_bullish/len(ge5_posts)*100:.1f}%)")
        print(f"  Bearish: {ge5_bearish:,} ({ge5_bearish/len(ge5_posts)*100:.1f}%)")
        
        # Posts with ≥10 emojis
        ge10_posts = df[df['unique_emoji_count'] >= 10]
        ge10_bullish = len(ge10_posts[ge10_posts['label'] == 'bullish'])
        ge10_bearish = len(ge10_posts[ge10_posts['label'] == 'bearish'])
        
        print(f"\nPosts with ≥10 emojis: {len(ge10_posts):,} ({len(ge10_posts)/total_posts*100:.1f}%)")
        print(f"  Bullish: {ge10_bullish:,} ({ge10_bullish/len(ge10_posts)*100:.1f}%)")
        print(f"  Bearish: {ge10_bearish:,} ({ge10_bearish/len(ge10_posts)*100:.1f}%)")
        
        # Emoji count statistics
        print(f"\nEmoji count statistics:")
        print(f"  Mean unique emojis per post: {df['unique_emoji_count'].mean():.2f}")
        print(f"  Median unique emojis per post: {df['unique_emoji_count'].median():.2f}")
        print(f"  Max unique emojis per post: {df['unique_emoji_count'].max()}")
        
        # Find the bin with highest bullish ratio
        max_bullish_ratio_idx = results_df['bullish_ratio'].idxmax()
        max_bullish_ratio_bin = results_df.loc[max_bullish_ratio_idx, 'bin_label']
        max_bullish_ratio = results_df.loc[max_bullish_ratio_idx, 'bullish_ratio']
        
        print(f"\nHighest bullish ratio: {max_bullish_ratio:.4f} in bin '{max_bullish_ratio_bin}'")
        
        # Find the bin with lowest bullish ratio
        min_bullish_ratio_idx = results_df['bullish_ratio'].idxmin()
        min_bullish_ratio_bin = results_df.loc[min_bullish_ratio_idx, 'bin_label']
        min_bullish_ratio = results_df.loc[min_bullish_ratio_idx, 'bullish_ratio']
        
        print(f"Lowest bullish ratio: {min_bullish_ratio:.4f} in bin '{min_bullish_ratio_bin}'")

    def run_analysis(self):
        """Run the complete emoji count sentiment analysis"""
        print("Starting Emoji Count vs Sentiment Analysis")
        print("=" * 80)
        
        # Load and process data
        df = self.load_and_process_data()
        
        # Analyze distribution
        results_df = self.analyze_emoji_count_distribution(df)
        
        # Create visualization
        self.create_visualization(results_df)
        
        # Save results
        self.save_results(results_df)
        
        # Generate summary statistics
        self.generate_summary_statistics(df, results_df)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description="Analyze emoji count vs sentiment distribution")
    parser.add_argument("--dataset", required=True, help="Path to dataset CSV file")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    args = parser.parse_args()
    
    # Create analyzer instance
    analyzer = EmojiCountSentimentAnalyzer(args.dataset, args.output_dir)
    
    # Run analysis
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
