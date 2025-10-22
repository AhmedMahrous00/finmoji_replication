import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
import time
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import csr_matrix

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

def lr_work_proxy_train(X_csr, n_iter, epochs=1):
    return X_csr.nnz * max(1, int(n_iter)) * epochs

def lr_work_proxy_infer(X_csr):
    return X_csr.nnz

def lr_params(model):
    return int(model.coef_.size + model.intercept_.size)

def transformer_params(hf_model):
    return sum(p.numel() for p in hf_model.parameters())

def transformer_forward_macs_per_layer(seq_len, d_model, d_ff):
    return 4*seq_len*d_model*d_model + 2*(seq_len**2)*d_model + 2*seq_len*d_model*d_ff

def transformer_work_infer_per_sample(seq_len, layers, d_model, d_ff):
    return layers * transformer_forward_macs_per_layer(seq_len, d_model, d_ff)

def transformer_work_train_per_sample(seq_len, layers, d_model, d_ff):
    return 2 * transformer_work_infer_per_sample(seq_len, layers, d_model, d_ff)

def transformer_work_infer_total(seq_len, layers, d_model, d_ff, n_samples):
    return n_samples * transformer_work_infer_per_sample(seq_len, layers, d_model, d_ff)

def transformer_work_train_total(seq_len, layers, d_model, d_ff, n_train_samples, epochs=1):
    return epochs * n_train_samples * transformer_work_train_per_sample(seq_len, layers, d_model, d_ff)

def tokens_processed_train(seq_len, batch_size, steps):
    return seq_len * batch_size * steps

def evaluate_lr_efficiency(dataset_dir, output_dir):
    datasets = {
        'Text only': f'{dataset_dir}/text_only.csv',
        'Emojis only': f'{dataset_dir}/emoji_only.csv', 
        'Text + Emojis': f'{dataset_dir}/text_emoji.csv'
    }
    
    results = {}
    
    for name, filepath in datasets.items():
        print(f"Evaluating LR efficiency for {name}...")
        
        df = pd.read_csv(filepath)
        
        train_ids = pd.read_csv(f'{dataset_dir}/splits/train_ids.txt', header=None)[0].tolist()
        test_ids = pd.read_csv(f'{dataset_dir}/splits/test_ids.txt', header=None)[0].tolist()
        
        train_df = df[df['post_id'].isin(train_ids)]
        test_df = df[df['post_id'].isin(test_ids)]
        
        # Use exactly 80k for training and 20k for inference
        train_df = train_df.sample(n=min(80000, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(20000, len(test_df)), random_state=42)
        
        text_col = 'emojis' if 'emojis' in df.columns else 'text'
        
        train_df = train_df.dropna(subset=[text_col]).copy()
        test_df = test_df.dropna(subset=[text_col]).copy()
        train_df[text_col] = train_df[text_col].fillna('').astype(str)
        test_df[text_col] = test_df[text_col].fillna('').astype(str)
        
        train_df = train_df[train_df[text_col].str.strip() != ''].copy()
        test_df = test_df[test_df[text_col].str.strip() != ''].copy()
        
        pattern = r'\w+|[^\s]'
        tokenizer = RegexpTokenizer(pattern)
        vectorizer = TfidfVectorizer(max_features=5000, min_df=1, tokenizer=tokenizer.tokenize)
        
        X_train = vectorizer.fit_transform(train_df[text_col])
        X_test = vectorizer.transform(test_df[text_col])
        y_train = train_df['label']
        y_test = test_df['label']
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        params = lr_params(model)
        train_work = lr_work_proxy_train(X_train, model.n_iter_[0], epochs=1)
        infer_work = lr_work_proxy_infer(X_test)
        
        results[name] = {
            'model_type': 'Logistic Regression',
            'parameters': params,
            'train_work': train_work,
            'infer_work': infer_work,
            'train_work_per_sample': train_work / len(train_df),
            'infer_work_per_sample': infer_work / len(test_df),
            'n_features': X_train.shape[1],
            'n_train_samples': len(train_df),
            'n_test_samples': len(test_df),
            'n_iterations': model.n_iter_[0]
        }
        
        print(f"  Parameters: {params:,}")
        print(f"  Train work: {train_work:,.0f} MACs")
        print(f"  Infer work: {infer_work:,.0f} MACs")
        print(f"  Train work/sample: {train_work / len(train_df):,.0f} MACs")
        print(f"  Infer work/sample: {infer_work / len(test_df):,.0f} MACs")
    
    return results

def evaluate_transformer_efficiency(dataset_dir, output_dir):
    datasets = {
        'Text only': f'{dataset_dir}/text_only.csv',
        'Emojis only': f'{dataset_dir}/emoji_only.csv', 
        'Text + Emojis': f'{dataset_dir}/text_emoji.csv'
    }
    
    results = {}
    
    for name, filepath in datasets.items():
        print(f"Evaluating Transformer efficiency for {name}...")
        
        df = pd.read_csv(filepath)
        
        text_col = 'emojis' if 'emojis' in df.columns else 'text'
        
        df = df.dropna(subset=[text_col, 'label']).copy()
        df[text_col] = df[text_col].fillna('').astype(str)
        df = df[df[text_col].str.strip() != ''].copy()
        
        # Use exactly 80k for training and 20k for inference
        train_df = df.sample(n=min(80000, len(df)), random_state=42)
        test_df = df.sample(n=min(20000, len(df)), random_state=42)
        
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        def map_labels(labels):
            return [0 if label == 'bearish' else 2 if label == 'bullish' else 1 for label in labels]
        
        train_encodings = tokenizer(train_df[text_col].tolist(), truncation=True, padding=True, max_length=120, return_tensors="pt")
        test_encodings = tokenizer(test_df[text_col].tolist(), truncation=True, padding=True, max_length=120, return_tensors="pt")
        
        train_labels = map_labels(train_df['label'].tolist())
        test_labels = map_labels(test_df['label'].tolist())
        train_dataset = Dataset(train_encodings, train_labels)
        test_dataset = Dataset(test_encodings, test_labels)
        
        config = model.config
        layers = config.num_hidden_layers
        d_model = config.hidden_size
        d_ff = config.intermediate_size
        heads = config.num_attention_heads
        
        seq_len = 120
        batch_size = 256
        steps = len(train_dataset) // batch_size
        
        params = transformer_params(model)
        
        # Use the corrected MACs calculation
        infer_work_per_sample = transformer_work_infer_per_sample(seq_len, layers, d_model, d_ff)
        train_work_per_sample = transformer_work_train_per_sample(seq_len, layers, d_model, d_ff)
        
        train_work = transformer_work_train_total(seq_len, layers, d_model, d_ff, len(train_df), epochs=1)
        infer_work = transformer_work_infer_total(seq_len, layers, d_model, d_ff, len(test_df))
        tokens_processed = seq_len * len(train_df)
        
        results[name] = {
            'model_type': 'Transformer (Twitter-RoBERTa)',
            'parameters': params,
            'train_work': train_work,
            'infer_work': infer_work,
            'train_work_per_sample': train_work_per_sample,
            'infer_work_per_sample': infer_work_per_sample,
            'tokens_processed_train': tokens_processed,
            'n_train_samples': len(train_df),
            'n_test_samples': len(test_df),
            'seq_len': seq_len,
            'batch_size': batch_size,
            'steps': steps,
            'layers': layers,
            'd_model': d_model,
            'd_ff': d_ff,
            'heads': heads
        }
        
        print(f"  Parameters: {params:,}")
        print(f"  Train work: {train_work:,.0f} MACs")
        print(f"  Infer work: {infer_work:,.0f} MACs")
        print(f"  Train work/sample: {train_work_per_sample:,.0f} MACs")
        print(f"  Infer work/sample: {infer_work_per_sample:,.0f} MACs")
        print(f"  Train:Infer ratio: {train_work_per_sample / infer_work_per_sample:.1f}x")
        print(f"  Tokens processed (train): {tokens_processed:,}")
    
    return results

def create_efficiency_table(lr_results, transformer_results, output_dir):
    print("\n" + "="*120)
    print("COMPUTATIONAL EFFICIENCY ANALYSIS")
    print("="*120)
    
    all_results = {}
    
    for dataset in ['Text only', 'Emojis only', 'Text + Emojis']:
        all_results[dataset] = {
            'LR': lr_results.get(dataset, {}),
            'Transformer': transformer_results.get(dataset, {})
        }
    
    print(f"{'Metric':<25} {'LR Text':>18} {'LR Emojis':>18} {'LR Text+Emojis':>20} {'TF Text':>18} {'TF Emojis':>18} {'TF Text+Emojis':>20}")
    print("-" * 140)
    
    metrics = [
        ('Parameters', 'parameters'),
        ('Train Work (MACs)', 'train_work'),
        ('Infer Work (MACs)', 'infer_work'),
        ('Train Work/Sample', 'train_work_per_sample'),
        ('Infer Work/Sample', 'infer_work_per_sample')
    ]
    
    for metric_name, metric_key in metrics:
        row = f"{metric_name:<25}"
        for dataset in ['Text only', 'Emojis only', 'Text + Emojis']:
            for model_type in ['LR', 'Transformer']:
                if dataset in all_results and model_type in all_results[dataset]:
                    value = all_results[dataset][model_type].get(metric_key, 0)
                    if metric_key in ['parameters', 'train_work', 'infer_work']:
                        row += f"{value:>18,}"
                    else:
                        row += f"{value:>18,.0f}"
                else:
                    row += f"{'N/A':>18}"
        print(row)
    
    print("="*140)
    
    csv_path = Path(output_dir) / "computational_efficiency_results.csv"
    
    rows = []
    for dataset in ['Text only', 'Emojis only', 'Text + Emojis']:
        for model_type in ['LR', 'Transformer']:
            if dataset in all_results and model_type in all_results[dataset]:
                result = all_results[dataset][model_type]
                row = {
                    'dataset': dataset,
                    'model_type': model_type,
                    'parameters': result.get('parameters', 0),
                    'train_work': result.get('train_work', 0),
                    'infer_work': result.get('infer_work', 0),
                    'train_work_per_sample': result.get('train_work_per_sample', 0),
                    'infer_work_per_sample': result.get('infer_work_per_sample', 0),
                    'n_train_samples': result.get('n_train_samples', 0),
                    'n_test_samples': result.get('n_test_samples', 0)
                }
                if model_type == 'Transformer':
                    row.update({
                        'tokens_processed_train': result.get('tokens_processed_train', 0),
                        'seq_len': result.get('seq_len', 0),
                        'batch_size': result.get('batch_size', 0),
                        'steps': result.get('steps', 0),
                        'layers': result.get('layers', 0),
                        'd_model': result.get('d_model', 0),
                        'd_ff': result.get('d_ff', 0),
                        'heads': result.get('heads', 0)
                    })
                rows.append(row)
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    latex_path = Path(output_dir) / "computational_efficiency_table.txt"
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Computational Efficiency Analysis. This table presents device-agnostic metrics for Logistic Regression and Transformer models, including parameter counts and work proxies (MACs) for training and inference. Work proxies scale with data size, sequence length, and model complexity.}\n")
        f.write("\\label{tab:computational-efficiency}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Metric} & \\textbf{LR Text} & \\textbf{LR Emojis} & \\textbf{LR Text+Emojis} & \\textbf{TF Text} & \\textbf{TF Emojis} & \\textbf{TF Text+Emojis} \\\\\n")
        f.write("\\midrule\n")
        
        for metric_name, metric_key in metrics:
            row = f"{metric_name}"
            for dataset in ['Text only', 'Emojis only', 'Text + Emojis']:
                for model_type in ['LR', 'Transformer']:
                    if dataset in all_results and model_type in all_results[dataset]:
                        value = all_results[dataset][model_type].get(metric_key, 0)
                        if metric_key in ['parameters', 'train_work', 'infer_work']:
                            row += f"&{value:,}"
                        else:
                            row += f"&{value:.0f}"
                    else:
                        row += "&N/A"
            row += "\\\\\n"
            f.write(row)
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {latex_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing dataset files")
    parser.add_argument("--output-dir", default="replication/results/plots", help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Evaluating Logistic Regression computational efficiency...")
    lr_results = evaluate_lr_efficiency(args.dataset_dir, args.output_dir)
    
    print("\nEvaluating Transformer computational efficiency...")
    transformer_results = evaluate_transformer_efficiency(args.dataset_dir, args.output_dir)
    
    print("\nCreating efficiency comparison table...")
    create_efficiency_table(lr_results, transformer_results, args.output_dir)
    
    print("\nComputational efficiency analysis completed!")

if __name__ == "__main__":
    main()
