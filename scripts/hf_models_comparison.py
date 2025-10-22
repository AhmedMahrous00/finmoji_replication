import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import time
from pathlib import Path
import emoji

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

def evaluate_all_hf_models(dataset_dir, output_dir):
    # Define models to test
    models = {
        'Twitter-RoBERTa': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'FinBERT': 'ProsusAI/finbert',
        'DistilBERT': 'distilbert/distilbert-base-uncased',
        'FinancialBERT': 'ahmedrachid/FinancialBERT',
        'CryptoBERT': 'kk08/CryptoBERT',
        'FinTwitBERT': 'StephanAkkerman/FinTwitBERT-sentiment'
    }
    
    datasets = {
        'Text only': f'{dataset_dir}/text_only.csv',
        'Emojis only': f'{dataset_dir}/emoji_only.csv', 
        'Text + Emojis': f'{dataset_dir}/text_emoji.csv'
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*80}")
        
        results[model_name] = {}
        
        for dataset_name, filepath in datasets.items():
            print(f"\n  Processing {dataset_name}...")
            
            try:
                # Load data
                df = pd.read_csv(filepath)
                
                # Clean the data
                df = df.dropna(subset=['text', 'label']).copy()
                df['text'] = df['text'].fillna('').astype(str)
                df = df[df['text'].str.strip() != ''].copy()
                
                # Use 90% train, 10% test split
                train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
                
                print(f"    Train: {len(train_df)}, Test: {len(test_df)}")
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                
                # Handle emoji preprocessing for models that need it
                # Twitter-RoBERTa handles emojis natively, others need demojizing
                if model_name != 'Twitter-RoBERTa':
                    print(f"    Demojizing text for {model_name}...")
                    train_df = train_df.copy()
                    test_df = test_df.copy()
                    train_df['text'] = train_df['text'].apply(lambda x: emoji.demojize(x))
                    test_df['text'] = test_df['text'].apply(lambda x: emoji.demojize(x))
                
                # Map binary labels to model's expected format
                def map_labels(labels, model_name):
                    if model_name == 'Twitter-RoBERTa':
                        # Twitter-RoBERTa expects 3 classes: 0=negative, 1=neutral, 2=positive
                        return [0 if label == 'bearish' else 2 if label == 'bullish' else 1 for label in labels]
                    else:
                        # Other models expect 3 classes: 0=negative, 1=neutral, 2=positive
                        return [0 if label == 'bearish' else 2 if label == 'bullish' else 1 for label in labels]
                
                # Tokenize with 120 tokens max length
                print(f"    Tokenizing {len(train_df)} training samples...")
                train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=120, return_tensors="pt")
                print(f"    Tokenizing {len(test_df)} test samples...")
                test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=120, return_tensors="pt")
                
                # Create datasets
                train_labels = map_labels(train_df['label'].tolist(), model_name)
                test_labels = map_labels(test_df['label'].tolist(), model_name)
                train_dataset = Dataset(train_encodings, train_labels)
                test_dataset = Dataset(test_encodings, test_labels)
                
                # Training arguments optimized for A6000
                training_args = TrainingArguments(
                    output_dir=f'{output_dir}/hf_models/{model_name}_{dataset_name.replace(" ", "_").replace("+", "_")}',
                    num_train_epochs=3,
                    per_device_train_batch_size=128,
                    per_device_eval_batch_size=256,
                    warmup_steps=50,
                    weight_decay=0.01,
                    logging_dir=f'{output_dir}/logs/hf_models/{model_name}_{dataset_name.replace(" ", "_").replace("+", "_")}',
                    logging_steps=100,
                    eval_strategy="no",
                    save_strategy="no",
                    fp16=True,
                    dataloader_pin_memory=True,
                    dataloader_num_workers=32,
                    remove_unused_columns=True,
                    gradient_accumulation_steps=1,
                    dataloader_drop_last=False,
                    max_grad_norm=1.0,
                    optim="adamw_torch_fused",
                    lr_scheduler_type="linear",
                    learning_rate=2e-5,
                    dataloader_prefetch_factor=4,
                )
                
                # Data collator
                data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
                
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                    data_collator=data_collator,
                )
                
                # Train the model
                print(f"    Training {model_name} on {dataset_name}...")
                trainer.train()
                
                # Evaluate
                predictions = trainer.predict(test_dataset)
                y_pred_3class = np.argmax(predictions.predictions, axis=1)
                
                # Map back to binary labels
                y_pred = ['bearish' if pred == 0 else 'bullish' for pred in y_pred_3class]
                y_true = test_df['label'].tolist()
                
                report = classification_report(y_true, y_pred, output_dict=True)
                f1_score = report['macro avg']['f1-score']
                
                results[model_name][dataset_name] = f1_score
                
                print(f"    {model_name} on {dataset_name}: F1 = {f1_score:.4f}")
                
            except Exception as e:
                print(f"    ERROR with {model_name} on {dataset_name}: {str(e)}")
                results[model_name][dataset_name] = 0.0
    
    create_hf_models_table(results, output_dir)
    return results

def create_hf_models_table(results, output_dir):
    print("\n" + "="*80)
    print("HUGGING FACE MODELS COMPARISON TABLE")
    print("="*80)
    
    # Create table
    print(f"{'Model':<20} {'Text only':>12} {'Emojis only':>12} {'Text + Emojis':>15}")
    print("-" * 80)
    
    for model_name in ['Twitter-RoBERTa', 'FinBERT', 'DistilBERT', 'FinancialBERT', 'CryptoBERT', 'FinTwitBERT']:
        if model_name in results:
            row = f"{model_name:<20}"
            for dataset in ['Text only', 'Emojis only', 'Text + Emojis']:
                if dataset in results[model_name]:
                    row += f"{results[model_name][dataset]:>12.2f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)
    
    print("="*80)
    
    # Save results to CSV
    results_df = pd.DataFrame(results).T
    csv_path = Path(output_dir) / "hf_models_comparison_results.csv"
    results_df.to_csv(csv_path)
    print(f"\nResults saved to {csv_path}")
    
    # Save LaTeX table
    latex_path = Path(output_dir) / "hf_models_comparison_table.txt"
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{F$_1$ Scores of Fine-tuned Transformer-based BERT Models. This table presents the F$_1$ scores achieved by various BERT models when trained on text only, emojis only, and text with emojis.}\n")
        f.write("\\label{table:huggingfacemodels}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("                    \\textbf{F1 Score} & \\textbf{Text only} & \\textbf{Emojis only} & \\textbf{Text + Emojis} \\\\ \\midrule\n")
        f.write("\\textbf{Model} &                   &                      &                        \\\\\n")
        
        for model_name in ['Twitter-RoBERTa', 'FinBERT', 'DistilBERT', 'FinancialBERT', 'CryptoBERT', 'FinTwitBERT']:
            if model_name in results:
                row = f"{model_name:<20}"
                for dataset in ['Text only', 'Emojis only', 'Text + Emojis']:
                    if dataset in results[model_name]:
                        row += f"&{results[model_name][dataset]:.2f:>8}"
                    else:
                        row += "&        "
                row += "\\\\\n"
                f.write(row)
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {latex_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing dataset files")
    parser.add_argument("--output-dir", default="replication/results/plots", help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate_all_hf_models(args.dataset_dir, args.output_dir)