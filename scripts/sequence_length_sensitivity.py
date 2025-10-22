import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
import re
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from bootstrap_utils import bootstrap_metrics, bootstrap_speed_metrics
import argparse
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class SequenceLengthSensitivityExperiment:
    def __init__(self, dataset_dir, output_dir, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.seq_lengths = [10, 20, 50, 120]
        
        # Load data splits
        self.train_ids = pd.read_csv(self.dataset_dir / 'splits/train_ids.txt', header=None)[0].tolist()
        self.test_ids = pd.read_csv(self.dataset_dir / 'splits/test_ids.txt', header=None)[0].tolist()
        
        # Load datasets
        self.datasets = {
            'text_only': pd.read_csv(self.dataset_dir / 'text_only.csv'),
            'emoji_only': pd.read_csv(self.dataset_dir / 'emoji_only.csv'),
            'text_emoji': pd.read_csv(self.dataset_dir / 'text_emoji.csv')
        }
        
        # Initialize tokenizer for transformer models
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Results storage
        self.results = {
            'lr_results': [],
            'transformer_results': [],
            'percentile_stats': None
        }

    def calculate_percentile_stats(self):
        """Calculate percentile statistics for text and emoji lengths"""
        print("Calculating percentile statistics...")
        
        # Use text_emoji dataset for comprehensive analysis
        df = self.datasets['text_emoji']
        
        # Calculate text lengths (character count)
        df['text_length'] = df['text'].str.len()
        
        # Calculate emoji lengths
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+"
        )
        df['emojis_extracted'] = df['text'].apply(lambda x: emoji_pattern.findall(str(x)))
        df['emoji_text'] = df['emojis_extracted'].apply(lambda x: ''.join(x))
        df['emoji_length'] = df['emoji_text'].str.len()
        
        # Calculate token lengths using the transformer tokenizer
        df['token_length'] = df['text'].apply(lambda x: len(self.tokenizer.encode(str(x), truncation=True, max_length=512)))
        
        # Filter out empty posts
        df = df[(df['text_length'] > 0) | (df['emoji_length'] > 0)]
        
        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 90, 95, 99]
        
        text_percentiles = np.percentile(df['text_length'], percentiles)
        emoji_percentiles = np.percentile(df['emoji_length'], percentiles)
        token_percentiles = np.percentile(df['token_length'], percentiles)
        
        # Create results table
        results_data = {
            'Percentile': ['5th', '25th', '50th', '75th', '90th', '95th', '99th'],
            'Text Length (chars)': [round(val, 2) for val in text_percentiles],
            'Emoji Length (chars)': [round(val, 2) for val in emoji_percentiles],
            'Token Length': [round(val, 2) for val in token_percentiles]
        }
        
        results_df = pd.DataFrame(results_data)
        
        # Save percentile table
        percentile_path = self.output_dir / "sequence_length_percentiles.csv"
        results_df.to_csv(percentile_path, index=False)
        
        print(f"Percentile statistics saved to {percentile_path}")
        print("\nPercentile Distribution:")
        print(results_df.to_string(index=False))
        
        self.results['percentile_stats'] = results_df
        return results_df

    def evaluate_lr_model(self, dataset_name, seq_length):
        """Evaluate Logistic Regression model with sequence length constraint"""
        print(f"Evaluating LR {dataset_name} with max length {seq_length}...")
        
        df = self.datasets[dataset_name]
        
        # Get train/test splits
        train_df = df[df['post_id'].isin(self.train_ids)]
        test_df = df[df['post_id'].isin(self.test_ids)]
        
        # Determine text column
        text_col = 'emojis' if 'emojis' in df.columns else 'text'
        
        # Clean and filter data
        train_df = train_df.dropna(subset=[text_col]).copy()
        test_df = test_df.dropna(subset=[text_col]).copy()
        train_df[text_col] = train_df[text_col].fillna('').astype(str)
        test_df[text_col] = test_df[text_col].fillna('').astype(str)
        
        # Apply sequence length constraint
        def truncate_text(text, max_length):
            if len(str(text)) <= max_length:
                return str(text)
            return str(text)[:max_length]
        
        train_df[text_col] = train_df[text_col].apply(lambda x: truncate_text(x, seq_length))
        test_df[text_col] = test_df[text_col].apply(lambda x: truncate_text(x, seq_length))
        
        # Remove empty texts
        train_df = train_df[train_df[text_col].str.strip() != ''].copy()
        test_df = test_df[test_df[text_col].str.strip() != ''].copy()
        
        # Vectorize text
        pattern = r'\w+|[^\s]'
        tokenizer = RegexpTokenizer(pattern)
        vectorizer = TfidfVectorizer(max_features=5000, min_df=1, tokenizer=tokenizer.tokenize)
        
        X_train = vectorizer.fit_transform(train_df[text_col])
        X_test = vectorizer.transform(test_df[text_col])
        
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        # Train model
        start_time = time.time()
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predict
        start_time = time.time()
        y_pred = lr_model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        
        # Bootstrap confidence intervals
        bootstrap_results = bootstrap_metrics(y_test, y_pred, B=100)
        
        # Calculate MACs (approximate)
        n_features = X_train.shape[1]
        n_samples = len(y_test)
        macs = n_features * n_samples  # Approximate MACs for inference
        
        return {
            'model_type': 'Logistic Regression',
            'dataset': dataset_name,
            'seq_length': seq_length,
            'f1_score': f1,
            'f1_ci_low': bootstrap_results['f1']['ci_low'],
            'f1_ci_high': bootstrap_results['f1']['ci_high'],
            'training_time': training_time,
            'inference_time': inference_time,
            'macs': macs,
            'n_features': n_features,
            'n_test_samples': len(y_test)
        }

    def evaluate_transformer_model(self, dataset_name, seq_length):
        """Evaluate Transformer model with sequence length constraint"""
        print(f"Evaluating Transformer {dataset_name} with max length {seq_length}...")
        
        df = self.datasets[dataset_name]
        
        # Get train/test splits
        train_df = df[df['post_id'].isin(self.train_ids)]
        test_df = df[df['post_id'].isin(self.test_ids)]
        
        # Determine text column
        text_col = 'emojis' if 'emojis' in df.columns else 'text'
        
        # Clean and filter data
        train_df = train_df.dropna(subset=[text_col]).copy()
        test_df = test_df.dropna(subset=[text_col]).copy()
        train_df[text_col] = train_df[text_col].fillna('').astype(str)
        test_df[text_col] = test_df[text_col].fillna('').astype(str)
        
        # Remove empty texts
        train_df = train_df[train_df[text_col].str.strip() != ''].copy()
        test_df = test_df[test_df[text_col].str.strip() != ''].copy()
        
        # Tokenize with sequence length constraint
        def tokenize_texts(texts, max_length):
            return self.tokenizer(
                texts.tolist(),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            )
        
        train_encodings = tokenize_texts(train_df[text_col], seq_length)
        test_encodings = tokenize_texts(test_df[text_col], seq_length)
        
        # Create datasets
        train_dataset = Dataset(train_encodings, train_df['label'].values)
        test_dataset = Dataset(test_encodings, test_df['label'].values)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f'./temp_model_{dataset_name}_{seq_length}',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'./logs_{dataset_name}_{seq_length}',
            logging_steps=10,
            evaluation_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            report_to=None,
            seed=42
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )
        
        # Train model
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        # Evaluate
        start_time = time.time()
        predictions = trainer.predict(test_dataset)
        inference_time = time.time() - start_time
        
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_test = test_df['label'].values
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
        
        # Bootstrap confidence intervals
        bootstrap_results = bootstrap_metrics(y_test, y_pred, B=100)
        
        # Calculate MACs (approximate for transformer)
        # Approximate MACs = 2 * seq_length * hidden_size * num_layers * batch_size
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers
        batch_size = 16
        macs = 2 * seq_length * hidden_size * num_layers * batch_size * len(y_test)
        
        # Clean up temporary files
        import shutil
        if os.path.exists(f'./temp_model_{dataset_name}_{seq_length}'):
            shutil.rmtree(f'./temp_model_{dataset_name}_{seq_length}')
        if os.path.exists(f'./logs_{dataset_name}_{seq_length}'):
            shutil.rmtree(f'./logs_{dataset_name}_{seq_length}')
        
        return {
            'model_type': 'Transformer',
            'dataset': dataset_name,
            'seq_length': seq_length,
            'f1_score': f1,
            'f1_ci_low': bootstrap_results['f1']['ci_low'],
            'f1_ci_high': bootstrap_results['f1']['ci_high'],
            'training_time': training_time,
            'inference_time': inference_time,
            'macs': macs,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'n_test_samples': len(y_test)
        }

    def run_lr_experiments(self):
        """Run Logistic Regression experiments across all datasets and sequence lengths"""
        print("=" * 80)
        print("RUNNING LOGISTIC REGRESSION EXPERIMENTS")
        print("=" * 80)
        
        for dataset_name in ['text_only', 'emoji_only', 'text_emoji']:
            for seq_length in self.seq_lengths:
                result = self.evaluate_lr_model(dataset_name, seq_length)
                self.results['lr_results'].append(result)
                print(f"LR {dataset_name} (length {seq_length}): F1 = {result['f1_score']:.4f}")

    def run_transformer_experiments(self):
        """Run Transformer experiments across all datasets and sequence lengths"""
        print("=" * 80)
        print("RUNNING TRANSFORMER EXPERIMENTS")
        print("=" * 80)
        
        for dataset_name in ['text_only', 'emoji_only', 'text_emoji']:
            for seq_length in self.seq_lengths:
                result = self.evaluate_transformer_model(dataset_name, seq_length)
                self.results['transformer_results'].append(result)
                print(f"Transformer {dataset_name} (length {seq_length}): F1 = {result['f1_score']:.4f}")

    def create_visualizations(self):
        """Create publication-quality visualizations"""
        print("Creating visualizations...")
        
        # Convert results to DataFrames
        lr_df = pd.DataFrame(self.results['lr_results'])
        transformer_df = pd.DataFrame(self.results['transformer_results'])
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sequence Length Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # F1 Score vs Sequence Length
        ax1 = axes[0, 0]
        for dataset in ['text_only', 'emoji_only', 'text_emoji']:
            lr_data = lr_df[lr_df['dataset'] == dataset]
            ax1.plot(lr_data['seq_length'], lr_data['f1_score'], 
                    marker='o', linewidth=2, label=f'LR {dataset}')
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Logistic Regression: F1 vs Sequence Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for dataset in ['text_only', 'emoji_only', 'text_emoji']:
            trans_data = transformer_df[transformer_df['dataset'] == dataset]
            ax2.plot(trans_data['seq_length'], trans_data['f1_score'], 
                    marker='s', linewidth=2, label=f'Transformer {dataset}')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Transformer: F1 vs Sequence Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Inference Time vs Sequence Length
        ax3 = axes[1, 0]
        for dataset in ['text_only', 'emoji_only', 'text_emoji']:
            lr_data = lr_df[lr_df['dataset'] == dataset]
            ax3.plot(lr_data['seq_length'], lr_data['inference_time'], 
                    marker='o', linewidth=2, label=f'LR {dataset}')
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Inference Time (seconds)')
        ax3.set_title('Logistic Regression: Inference Time vs Sequence Length')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        for dataset in ['text_only', 'emoji_only', 'text_emoji']:
            trans_data = transformer_df[transformer_df['dataset'] == dataset]
            ax4.plot(trans_data['seq_length'], trans_data['inference_time'], 
                    marker='s', linewidth=2, label=f'Transformer {dataset}')
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Inference Time (seconds)')
        ax4.set_title('Transformer: Inference Time vs Sequence Length')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "sequence_length_sensitivity_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {plot_path}")
        plt.show()
        
        # Create MACs vs F1 plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot LR results
        for dataset in ['text_only', 'emoji_only', 'text_emoji']:
            lr_data = lr_df[lr_df['dataset'] == dataset]
            ax.scatter(lr_data['macs'], lr_data['f1_score'], 
                      s=100, alpha=0.7, label=f'LR {dataset}', marker='o')
        
        # Plot Transformer results
        for dataset in ['text_only', 'emoji_only', 'text_emoji']:
            trans_data = transformer_df[transformer_df['dataset'] == dataset]
            ax.scatter(trans_data['macs'], trans_data['f1_score'], 
                      s=100, alpha=0.7, label=f'Transformer {dataset}', marker='s')
        
        ax.set_xlabel('MACs (Multiply-Accumulate Operations)')
        ax.set_ylabel('F1 Score')
        ax.set_title('Computational Efficiency: MACs vs F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        
        # Save MACs plot
        macs_plot_path = self.output_dir / "computational_efficiency_analysis.png"
        plt.savefig(macs_plot_path, dpi=300, bbox_inches='tight')
        print(f"MACs visualization saved to {macs_plot_path}")
        plt.show()

    def save_results(self):
        """Save all results to CSV files"""
        print("Saving results...")
        
        # Save LR results
        lr_df = pd.DataFrame(self.results['lr_results'])
        lr_path = self.output_dir / "lr_sequence_length_results.csv"
        lr_df.to_csv(lr_path, index=False)
        print(f"LR results saved to {lr_path}")
        
        # Save Transformer results
        transformer_df = pd.DataFrame(self.results['transformer_results'])
        transformer_path = self.output_dir / "transformer_sequence_length_results.csv"
        transformer_df.to_csv(transformer_path, index=False)
        print(f"Transformer results saved to {transformer_path}")
        
        # Save combined results
        combined_df = pd.concat([lr_df, transformer_df], ignore_index=True)
        combined_path = self.output_dir / "combined_sequence_length_results.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined results saved to {combined_path}")
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        print("\nLogistic Regression Results:")
        print(lr_df.groupby(['dataset', 'seq_length'])['f1_score'].mean().round(4))
        
        print("\nTransformer Results:")
        print(transformer_df.groupby(['dataset', 'seq_length'])['f1_score'].mean().round(4))
        
        # Find best performing configurations
        print("\nBest LR Configuration:")
        best_lr = lr_df.loc[lr_df['f1_score'].idxmax()]
        print(f"Dataset: {best_lr['dataset']}, Length: {best_lr['seq_length']}, F1: {best_lr['f1_score']:.4f}")
        
        print("\nBest Transformer Configuration:")
        best_trans = transformer_df.loc[transformer_df['f1_score'].idxmax()]
        print(f"Dataset: {best_trans['dataset']}, Length: {best_trans['seq_length']}, F1: {best_trans['f1_score']:.4f}")

    def run_experiment(self):
        """Run the complete sequence length sensitivity experiment"""
        print("Starting Sequence Length Sensitivity Experiment")
        print("=" * 80)
        
        # Calculate percentile statistics
        self.calculate_percentile_stats()
        
        # Run experiments
        self.run_lr_experiments()
        self.run_transformer_experiments()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Sequence Length Sensitivity Experiment")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", required=True, help="Path to output directory")
    parser.add_argument("--model-name", default="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                       help="Transformer model name")
    args = parser.parse_args()
    
    # Create experiment instance
    experiment = SequenceLengthSensitivityExperiment(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    # Run experiment
    experiment.run_experiment()

if __name__ == "__main__":
    main()
