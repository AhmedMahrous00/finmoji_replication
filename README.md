# Emoji Paper Replication

Replication package for the emoji sentiment analysis paper.
Runs core analyses focusing on data processing, model evaluation, and learning curves.

## What This Includes

This replication runs the analysis steps:

1. **Data Processing** - Builds datasets from raw parquet/CSV files
2. **Logistic Regression Models** - Comprehensive evaluation of baseline models
3. **Transformer Models** - Deep learning model evaluation and comparison
4. **Learning Curves** - Analysis of model performance vs training data size
5. **Tokenizer Audit** - Analysis of tokenization behavior and efficiency

## Quick Start

```bash
pip install -r requirements.txt
python launch_replication.py
```

Make sure output.parquet and emojitweets.csv are in the data/ folder before running.

**Estimated runtime:** 30-60 minutes

## License

MIT License. See LICENSE file.