# Emoji Paper Replication

Complete replication package for emoji sentiment analysis paper. Reproduces all analyses, visualizations, and results from raw data files.

## Quick Start

### Prerequisites
- Python 3.8+

### Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your data files
# Put output.parquet and emojitweets.csv in the data/ directory

# 3. Run replication
python launch_replication.py
```

## Usage

### Run Full Replication
```bash
python launch_replication.py
```
- **Auto-detects data files**
- **Interactive setup**
- **Runs all 9 analysis steps** (1-4 hours)

## Expected Runtime

| Analysis | Time |
|----------|------|
| Data Processing | 5-15 min |
| Logistic Regression | 10-30 min |
| Transformer Models | 30-90 min |
| Sequence Length | 20-60 min |
| Wordclouds | 2-5 min |
| Learning Curves | 15-45 min |
| **Total** | **1.5-4 hours** |

## Outputs

- **Datasets**: `text_only.csv`, `emoji_only.csv`, `text_emoji.csv`
- **Analysis Results**: Performance metrics, statistical summaries
- **Visualizations**: Emoji wordclouds, learning curves, efficiency charts
- **Reports**: Comprehensive execution logs

## Troubleshooting

### Memory Issues
```bash
export CUDA_VISIBLE_DEVICES=""
python launch_replication.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Large Datasets
The full replication handles large datasets automatically with optimized processing.

## License

MIT License - see [LICENSE](LICENSE) file for details.