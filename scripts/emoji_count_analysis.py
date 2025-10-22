import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from datetime import datetime

# Emoji pattern for detection
emoji_pattern = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF\u2700-\u27BF]"
)

def count_emojis(text):
    """Count total emojis in text"""
    if pd.isna(text):
        return 0
    return len(emoji_pattern.findall(str(text)))

def count_unique_emojis(text):
    """Count unique emojis in text (no repetition)"""
    if pd.isna(text):
        return 0
    emojis = emoji_pattern.findall(str(text))
    return len(set(emojis))

def create_emoji_count_bar_chart(input_file, output_dir):
    print(f"Loading data from {input_file}...")
    
    # Detect file format and load accordingly
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file, engine='pyarrow', columns=['text'])
    else:
        df = pd.read_csv(input_file, usecols=['text'])
    print(f"Loaded {len(df)} total rows")
    
    # Use all posts - no sampling
    
    # Convert text to string and count emojis
    print("Counting emojis in posts...")
    df['text'] = df['text'].astype(str)
    
    # Count emojis
    df['total_emojis'] = df['text'].apply(count_emojis)
    df['unique_emojis'] = df['text'].apply(count_unique_emojis)
    
    # Create bins for emoji counts (0, 1, 2, 3, ..., 9, 10+)
    df['total_emoji_bin'] = df['total_emojis'].apply(lambda x: min(x, 10))
    df['unique_emoji_bin'] = df['unique_emojis'].apply(lambda x: min(x, 10))
    
    # Count posts in each bin (including 0 emojis)
    total_counts = df['total_emoji_bin'].value_counts().sort_index()
    unique_counts = df['unique_emoji_bin'].value_counts().sort_index()
    
    # Convert counts to percentages
    total_percentages = (total_counts / len(df)) * 100
    unique_percentages = (unique_counts / len(df)) * 100
    
    print(f"Posts with emojis: {len(df[df['total_emojis'] > 0]):,}")
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(total_percentages))
    width = 0.35
    
    # Create labels for x-axis (0, 1, 2, ..., 9, 10+)
    labels = [str(i) if i <= 9 else '10+' for i in total_percentages.index]
    
    # Plot bars with percentages
    bars1 = ax.bar(x - width/2, total_percentages.values, width, label='Total Emojis (%)', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, unique_percentages.values, width, label='Unique Emojis (%)', color='lightcoral', alpha=0.8)
    
    # Customize the chart
    ax.set_xlabel('Number of Emojis per Post', fontsize=12)
    ax.set_ylabel('Percentage of Posts (%)', fontsize=12)
    ax.set_title('Distribution of Emoji Counts per Post (Percentages)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(total_percentages.max(), unique_percentages.max()) * 1.1)
    
    # Add value labels on bars (show percentages)
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / "emoji_count_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Emoji count bar chart saved to {plot_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\nEmoji Count Summary:")
    print(f"Total posts: {len(df):,}")
    print(f"Posts with emojis: {len(df[df['total_emojis'] > 0]):,}")
    print(f"Posts without emojis: {len(df[df['total_emojis'] == 0]):,} ({(len(df[df['total_emojis'] == 0])/len(df)*100):.2f}%)")
    print(f"Average total emojis per post (all posts): {df['total_emojis'].mean():.2f}")
    print(f"Average unique emojis per post (all posts): {df['unique_emojis'].mean():.2f}")
    print(f"Average total emojis per post (with emojis only): {df[df['total_emojis'] > 0]['total_emojis'].mean():.2f}")
    print(f"Average unique emojis per post (with emojis only): {df[df['total_emojis'] > 0]['unique_emojis'].mean():.2f}")
    print(f"Max total emojis in a single post: {df['total_emojis'].max()}")
    print(f"Max unique emojis in a single post: {df['unique_emojis'].max()}")
    
    # Print percentage breakdown
    print(f"\nPercentage breakdown by emoji count:")
    print(f"{'Count':<8} {'Total %':<10} {'Unique %':<10}")
    print("-" * 30)
    for i in range(0, 11):
        total_percent = total_percentages.get(i, 0)
        unique_percent = unique_percentages.get(i, 0)
        label = str(i) if i <= 9 else '10+'
        print(f"{label:<8} {total_percent:<10.2f} {unique_percent:<10.2f}")
    
    return df

def create_emoji_timeseries(input_file, output_dir):
    print(f"Loading data for time series analysis from {input_file}...")
    
    # Detect file format and load accordingly
    if input_file.endswith('.parquet'):
        try:
            df = pd.read_parquet(input_file, engine='pyarrow', columns=['text', 'created_at'])
        except:
            # If created_at doesn't exist, try other common timestamp column names
            try:
                df = pd.read_parquet(input_file, engine='pyarrow', columns=['text', 'timestamp'])
                df = df.rename(columns={'timestamp': 'created_at'})
            except:
                try:
                    df = pd.read_parquet(input_file, engine='pyarrow', columns=['text', 'date'])
                    df = df.rename(columns={'date': 'created_at'})
                except:
                    try:
                        df = pd.read_parquet(input_file, engine='pyarrow', columns=['text', 'time'])
                        df = df.rename(columns={'time': 'created_at'})
                    except:
    else:
        try:
            df = pd.read_csv(input_file, usecols=['text', 'created_at'])
        except:
            # If created_at doesn't exist, try other common timestamp column names
            try:
                df = pd.read_csv(input_file, usecols=['text', 'timestamp'])
                df = df.rename(columns={'timestamp': 'created_at'})
            except:
                try:
                    df = pd.read_csv(input_file, usecols=['text', 'date'])
                    df = df.rename(columns={'date': 'created_at'})
                except:
                    try:
                        df = pd.read_csv(input_file, usecols=['text', 'time'])
                        df = df.rename(columns={'time': 'created_at'})
                    except:
                        print("No timestamp column found. Creating dummy time series with sample data...")
                        return create_dummy_timeseries(output_dir)
    
    print(f"Loaded {len(df)} total rows")
    
    # Use all posts - no sampling
    
    # Convert timestamp to datetime
    try:
        # Try to convert - if values are Unix timestamps (numeric), specify unit='s'
        if pd.api.types.is_numeric_dtype(df['created_at']):
            df['created_at'] = pd.to_datetime(df['created_at'], unit='s')
        else:
            df['created_at'] = pd.to_datetime(df['created_at'])
        df['year_month'] = df['created_at'].dt.to_period('M')
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        print("Creating dummy time series...")
        return create_dummy_timeseries(output_dir)
    
    # Convert text to string and count emojis
    print("Counting emojis for time series...")
    df['text'] = df['text'].astype(str)
    df['emoji_count'] = df['text'].apply(count_emojis)
    
    # Show overall date range
    print(f"Overall date range in data: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"Total posts: {len(df):,}")
    print(f"Posts with emojis: {(df['emoji_count'] > 0).sum():,}")
    
    # Group by month (including all posts, not just those with emojis)
    monthly_stats = df.groupby('year_month').agg({
        'emoji_count': ['mean', 'count', lambda x: (x > 0).sum()]
    }).round(2)
    
    monthly_stats.columns = ['avg_emojis_per_post', 'total_posts', 'posts_with_emojis']
    monthly_stats = monthly_stats.reset_index()
    monthly_stats['year_month_str'] = monthly_stats['year_month'].astype(str)
    
    # Show date range of posts with emojis
    emoji_posts = df[df['emoji_count'] > 0]
    if len(emoji_posts) > 0:
        print(f"Date range for posts with emojis: {emoji_posts['created_at'].min()} to {emoji_posts['created_at'].max()}")
    
    # Create the time series plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot the time series
    ax.plot(monthly_stats['year_month_str'], monthly_stats['avg_emojis_per_post'], 
            marker='o', linewidth=2, markersize=6, color='darkblue')
    
    # Customize the chart
    ax.set_xlabel('Year-Month', fontsize=12)
    ax.set_ylabel('Average Emojis per Post', fontsize=12)
    ax.set_title('Average Number of Emojis Used per Post Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set x-axis to show all months but rotate labels for readability
    ax.set_xticks(range(len(monthly_stats)))
    ax.set_xticklabels(monthly_stats['year_month_str'], rotation=45, ha='right')
    
    # Adjust layout to accommodate rotated labels
    plt.subplots_adjust(bottom=0.15)
    
    # Add value labels on some points (every 6th point to avoid crowding but show more months)
    step = max(1, len(monthly_stats) // 20)  # Show up to 20 points
    for i in range(0, len(monthly_stats), step):
        ax.annotate(f"{monthly_stats.iloc[i]['avg_emojis_per_post']:.2f}",
                   (monthly_stats.iloc[i]['year_month_str'], monthly_stats.iloc[i]['avg_emojis_per_post']),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    # tight_layout() is replaced by subplots_adjust above
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / "emoji_usage_timeseries.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Emoji usage time series saved to {plot_path}")
    plt.show()
    
    # Save the data
    data_path = output_path / "monthly_emoji_stats.csv"
    monthly_stats.to_csv(data_path, index=False)
    print(f"Monthly emoji statistics saved to {data_path}")
    
    print(f"\nTime Series Summary:")
    print(f"Date range: {monthly_stats['year_month_str'].min()} to {monthly_stats['year_month_str'].max()}")
    print(f"Average emojis per post (overall): {monthly_stats['avg_emojis_per_post'].mean():.2f}")
    print(f"Highest average: {monthly_stats['avg_emojis_per_post'].max():.2f} in {monthly_stats.loc[monthly_stats['avg_emojis_per_post'].idxmax(), 'year_month_str']}")
    print(f"Lowest average: {monthly_stats['avg_emojis_per_post'].min():.2f} in {monthly_stats.loc[monthly_stats['avg_emojis_per_post'].idxmin(), 'year_month_str']}")
    
    return monthly_stats

def create_dummy_timeseries(output_dir):
    """Create a dummy time series when timestamp data is not available"""
    print("Creating dummy time series data...")
    
    # Generate dummy monthly data
    dates = pd.date_range('2020-01-01', '2024-12-01', freq='MS')
    dummy_data = []
    
    for date in dates:
        # Simulate some variation in emoji usage over time
        base_avg = 2.5
        seasonal_factor = 0.3 * np.sin(2 * np.pi * date.month / 12)
        trend_factor = 0.1 * (date.year - 2020)
        noise = np.random.normal(0, 0.2)
        
        avg_emojis = max(0.5, base_avg + seasonal_factor + trend_factor + noise)
        posts_count = np.random.randint(1000, 5000)
        
        dummy_data.append({
            'year_month': pd.Period(date, freq='M'),
            'avg_emojis_per_post': round(avg_emojis, 2),
            'posts_with_emojis': posts_count,
            'year_month_str': date.strftime('%Y-%m')
        })
    
    dummy_df = pd.DataFrame(dummy_data)
    
    # Create the dummy time series plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.plot(dummy_df['year_month_str'], dummy_df['avg_emojis_per_post'], 
            marker='o', linewidth=2, markersize=6, color='darkblue')
    
    ax.set_xlabel('Year-Month', fontsize=12)
    ax.set_ylabel('Average Emojis per Post', fontsize=12)
    ax.set_title('Average Number of Emojis Used per Post Over Time (Dummy Data)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / "emoji_usage_timeseries_dummy.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Dummy emoji usage time series saved to {plot_path}")
    plt.show()
    
    return dummy_df

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze emoji counts and create visualizations")
    parser.add_argument("--input", required=True, help="Path to input parquet file")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--plot-type", choices=['bar', 'timeseries', 'both'], default='both',
                       help="Type of plot to generate: 'bar' for bar chart only, 'timeseries' for line chart only, 'both' for both (default: both)")
    args = parser.parse_args()
    
    # Create emoji count bar chart
    if args.plot_type in ['bar', 'both']:
        emoji_posts = create_emoji_count_bar_chart(args.input, args.output_dir)
    
    # Create time series
    if args.plot_type in ['timeseries', 'both']:
        monthly_stats = create_emoji_timeseries(args.input, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
