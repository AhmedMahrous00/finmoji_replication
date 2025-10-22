import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from bootstrap_utils import bootstrap_metrics, bootstrap_speed_metrics

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)  # Pre-convert labels to tensor

    def __getitem__(self, idx):
        # More efficient tensor access
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx]
        }
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_all_transformer_models(dataset_dir, output_dir):
    datasets = {
        'Text only': f'{dataset_dir}/text_only.csv',
        'Emojis only': f'{dataset_dir}/emoji_only.csv', 
        'Text + Emojis': f'{dataset_dir}/text_emoji.csv'
    }
    
    results = {}
    confusion_matrices = {}
    
    for name, filepath in datasets.items():
        print(f"Evaluating {name}...")
        
        df = pd.read_csv(filepath)
        
        # Determine the text column name (could be 'text' or 'emojis')
        text_col = 'emojis' if 'emojis' in df.columns else 'text'
        
        # Clean the data - remove NaN values and ensure text is string
        df = df.dropna(subset=[text_col, 'label']).copy()
        df[text_col] = df[text_col].fillna('').astype(str)
        df = df[df[text_col].str.strip() != ''].copy()
        
        # Use 90% train, 10% test split (as specified in the paper)
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
        
        print(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # Map binary labels to 3-class format (bearish=0, bullish=2, neutral=1 unused)
        def map_labels(labels):
            return [0 if label == 'bearish' else 2 if label == 'bullish' else 1 for label in labels]
        
        # Tokenize with 120 tokens max length (as specified) - memory efficient
        print(f"  Tokenizing {len(train_df)} training samples...")
        train_encodings = tokenizer(train_df[text_col].tolist(), truncation=True, padding=True, max_length=120, return_tensors="pt")
        print(f"  Tokenizing {len(test_df)} test samples...")
        test_encodings = tokenizer(test_df[text_col].tolist(), truncation=True, padding=True, max_length=120, return_tensors="pt")
        
        # Create datasets before deleting dataframes
        train_labels = map_labels(train_df['label'].tolist())
        test_labels = map_labels(test_df['label'].tolist())
        train_dataset = Dataset(train_encodings, train_labels)
        test_dataset = Dataset(test_encodings, test_labels)
        
        # Store necessary info for speed testing before deletion
        train_size = len(train_df)
        test_size = len(test_df)
        train_texts = train_df[text_col].tolist()
        test_texts = test_df[text_col].tolist()
        train_label_list = train_df['label'].tolist()
        test_label_list = test_df['label'].tolist()
        
        # Free up memory by deleting the original dataframes
        del train_df, test_df, train_labels, test_labels
        import gc
        gc.collect()
        
        # Training arguments optimized for A6000 with 0.5TB RAM - MAX BATCH SIZE + 50 CORES
        training_args = TrainingArguments(
            output_dir=f'{output_dir}/transformer_{name.replace(" ", "_").replace("+", "_")}',
            num_train_epochs=3,
            per_device_train_batch_size=256,   # Maximum batch size for A6000 (48GB VRAM)
            per_device_eval_batch_size=256,    # Maximum eval batch size
            warmup_steps=50,  # 50 steps warmup as specified
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs/transformer_{name.replace(" ", "_").replace("+", "_")}',
            logging_steps=100,
            eval_strategy="no",
            save_strategy="no",
            fp16=True,  # Mixed precision training
            dataloader_pin_memory=True,   # Enable pin memory for faster data transfer
            dataloader_num_workers=32,    # High parallelism for 50 cores + 0.5TB RAM
            remove_unused_columns=True,
            gradient_accumulation_steps=1, # No gradient accumulation needed with large batch
            dataloader_drop_last=False,
            max_grad_norm=1.0,
            optim="adamw_torch_fused",    # Fused optimizer for A6000
            lr_scheduler_type="linear",
            learning_rate=2e-5,
            dataloader_prefetch_factor=4, # Prefetch 4 batches per worker for better GPU utilization
        )
        
        # Data collator for dynamic padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
        )
        
        # Speed benchmarking with specified sample sizes
        print("  Computing speed metrics...")
        
        # Training speed: 80,000 posts, 1 epoch
        speed_train_size = min(80000, train_size)
        # Sample from stored data
        import random
        random.seed(42)
        speed_indices = random.sample(range(train_size), speed_train_size)
        speed_train_texts = [train_texts[i] for i in speed_indices]
        speed_train_labels = [train_label_list[i] for i in speed_indices]
        speed_train_encodings = tokenizer(speed_train_texts, truncation=True, padding=True, max_length=120, return_tensors="pt")
        speed_train_dataset = Dataset(speed_train_encodings, map_labels(speed_train_labels))
        
        speed_trainer = Trainer(
            model=AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest"),
            args=TrainingArguments(
                output_dir=f'{output_dir}/speed_test',
                num_train_epochs=1,  # 1 epoch for speed test
                per_device_train_batch_size=256,  # Maximum batch for A6000
                logging_steps=100,
                eval_strategy="no",
                save_strategy="no",
                fp16=True,
                dataloader_pin_memory=True,
                dataloader_num_workers=32,  # High parallelism for 50 cores + 0.5TB RAM
                optim="adamw_torch_fused",
                dataloader_prefetch_factor=4,
            ),
            train_dataset=speed_train_dataset,
            data_collator=data_collator,
        )
        
        start_time = time.time()
        speed_trainer.train()
        training_time = time.time() - start_time
        
        # Inference speed: 20,000 posts
        speed_test_size = min(20000, test_size)
        # Sample from stored data
        speed_test_indices = random.sample(range(test_size), speed_test_size)
        speed_test_texts = [test_texts[i] for i in speed_test_indices]
        speed_test_labels = [test_label_list[i] for i in speed_test_indices]
        speed_test_encodings = tokenizer(speed_test_texts, truncation=True, padding=True, max_length=120, return_tensors="pt")
        speed_test_dataset = Dataset(speed_test_encodings, map_labels(speed_test_labels))
        
        start_time = time.time()
        predictions = speed_trainer.predict(speed_test_dataset)
        inference_time = time.time() - start_time
        
        # Full model training for accuracy metrics
        print("  Training full model for accuracy...")
        trainer.train()
        
        # Evaluate on full test set
        predictions = trainer.predict(test_dataset)
        y_pred_3class = np.argmax(predictions.predictions, axis=1)
        
        # Map back to binary labels (0->bearish, 2->bullish, 1->bullish if neutral)
        y_pred = ['bearish' if pred == 0 else 'bullish' for pred in y_pred_3class]
        y_true = test_label_list
        
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=['bearish', 'bullish'])
        
        # Compute bootstrap confidence intervals for accuracy metrics
        print(f"  Computing bootstrap confidence intervals for accuracy metrics...")
        bootstrap_results = bootstrap_metrics(y_true, y_pred, B=1000, rng=42)
        
        # Compute bootstrap confidence intervals for speed metrics
        print(f"  Computing bootstrap confidence intervals for speed metrics...")
        
        def train_func():
            # Use the speed trainer for consistent benchmarking
            start_time = time.time()
            speed_trainer.train()
            return time.time() - start_time
        
        def inference_func():
            # Use the speed trainer for consistent benchmarking
            start_time = time.time()
            speed_trainer.predict(speed_test_dataset)
            return time.time() - start_time
        
        speed_bootstrap_results = bootstrap_speed_metrics(train_func, inference_func, B=20, rng=42)
        
        results[name] = {
            'recall': report['macro avg']['recall'],
            'precision': report['macro avg']['precision'], 
            'f1': report['macro avg']['f1-score'],
            'training_time': training_time,
            'inference_time': inference_time,
            'bootstrap': bootstrap_results,
            'speed_bootstrap': speed_bootstrap_results
        }
        
        confusion_matrices[name] = {
            'matrix': cm,
            'y_test': y_true,
            'y_pred': y_pred
        }
        
        print(f"\n{'='*80}")
        print(f"COMPLETED: {name}")
        print(f"{'='*80}")
        print(f"  Recall:        {results[name]['recall']:.4f}")
        print(f"  Precision:     {results[name]['precision']:.4f}")
        print(f"  F1 Score:      {results[name]['f1']:.4f}")
        print(f"  Training time: {training_time:.2f}s ({training_time/60:.2f} min)")
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"{'='*80}\n")
    
    create_transformer_performance_table(results, output_dir)
    create_transformer_confusion_matrices(confusion_matrices, output_dir)
    
    return results, confusion_matrices

def create_transformer_performance_table(results, output_dir):
    print("\n" + "="*100)
    print("TRANSFORMER PERFORMANCE TABLE WITH BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*100)
    
    print(f"{'Metric':<20} {'Text only':>25} {'Emojis only':>25} {'Text + Emojis':>25}")
    print("-" * 100)
    
    print("\nAccuracy Metrics (Mean [95% CI]):")
    metrics = ['recall', 'precision', 'f1']
    metric_names = ['Recall', 'Precision', 'F1 Score']
    
    for metric, name in zip(metrics, metric_names):
        row = f"{name:<20}"
        for dataset in ['Text only', 'Emojis only', 'Text + Emojis']:
            mean_val = results[dataset]['bootstrap'][metric]['mean']
            ci_low = results[dataset]['bootstrap'][metric]['ci_low']
            ci_high = results[dataset]['bootstrap'][metric]['ci_high']
            row += f"{mean_val:>6.3f} [{ci_low:>5.3f}, {ci_high:>5.3f}]"
        print(row)
    
    print("\nSpeed Metrics (Mean [95% CI]):")
    speed_metrics = ['training_time', 'inference_time']
    speed_names = ['Training time (s)', 'Inference time (s)']
    
    for metric, name in zip(speed_metrics, speed_names):
        row = f"{name:<20}"
        for dataset in ['Text only', 'Emojis only', 'Text + Emojis']:
            mean_val = results[dataset]['speed_bootstrap'][metric]['mean']
            ci_low = results[dataset]['speed_bootstrap'][metric]['ci_low']
            ci_high = results[dataset]['speed_bootstrap'][metric]['ci_high']
            row += f"{mean_val:>6.0f} [{ci_low:>5.0f}, {ci_high:>5.0f}]"
        print(row)
    
    print("="*100)

def create_transformer_confusion_matrices(confusion_matrices, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    datasets = ['Emojis only', 'Text only', 'Text + Emojis']
    titles = ['(a) Emoji Only', '(b) Text Only', '(c) Text and Emoji']
    
    for i, (dataset, title) in enumerate(zip(datasets, titles)):
        cm = confusion_matrices[dataset]['matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Bearish', 'Bullish'], 
                    yticklabels=['Bearish', 'Bullish'])
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Predicted label')
        axes[i].set_ylabel('True label')
    
    plt.suptitle('Confusion Matrices of Transformer-based Twitter-roBERTa Models', 
                 y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = Path(output_dir) / "transformer_confusion_matrices.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing dataset files")
    parser.add_argument("--output-dir", default="replication/results/plots", help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate_all_transformer_models(args.dataset_dir, args.output_dir)
