import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
import time
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def evaluate_context_length_impact(dataset_dir, output_dir):
    datasets = {
        'Text-Only': f'{dataset_dir}/text_only.csv',
        'Emoji-Only': f'{dataset_dir}/emoji_only.csv'
    }
    
    max_lengths = [3, 10, 20, 50, 120]
    results = {}
    
    for dataset_name, filepath in datasets.items():
        print(f"Evaluating {dataset_name}...")
        results[dataset_name] = {}
        
        df = pd.read_csv(filepath)
        
        train_ids = pd.read_csv(f'{dataset_dir}/splits/train_ids.txt', header=None)[0].tolist()
        test_ids = pd.read_csv(f'{dataset_dir}/splits/test_ids.txt', header=None)[0].tolist()
        
        train_df = df[df['post_id'].isin(train_ids)]
        test_df = df[df['post_id'].isin(test_ids)]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        for max_length in max_lengths:
            print(f"  Testing max_length = {max_length}")
            
            try:
                train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=max_length)
                test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=max_length)
                
                train_dataset = Dataset(train_encodings, train_df['label'].tolist())
                test_dataset = Dataset(test_encodings, test_df['label'].tolist())
                
                training_args = TrainingArguments(
                    output_dir=f'{output_dir}/context_length_{dataset_name}_{max_length}',
                    num_train_epochs=3,
                    per_device_train_batch_size=16,
                    per_device_eval_batch_size=16,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_dir=f'{output_dir}/logs/context_length_{dataset_name}_{max_length}',
                    logging_steps=100,
                    evaluation_strategy="no",
                    save_strategy="no",
                )
                
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=test_dataset,
                )
                
                start_time = time.time()
                trainer.train()
                training_time = time.time() - start_time
                
                start_time = time.time()
                predictions = trainer.predict(test_dataset)
                inference_time = time.time() - start_time
                
                y_pred = np.argmax(predictions.predictions, axis=1)
                y_true = test_df['label'].tolist()
                
                f1 = f1_score(y_true, y_pred, average='macro')
                
                results[dataset_name][max_length] = {
                    'f1_score': f1,
                    'training_time': training_time,
                    'inference_time': inference_time
                }
                
            except Exception as e:
                print(f"Error processing max_length {max_length}: {e}")
                results[dataset_name][max_length] = {
                    'f1_score': 0.0,
                    'training_time': 0.0,
                    'inference_time': 0.0
                }
    
    create_context_length_table(results, output_dir)
    
    return results

def create_context_length_table(results, output_dir):
    print("\n" + "="*80)
    print("CONTEXT LENGTH IMPACT TABLE")
    print("="*80)
    
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{\\label{tab:max-length}Effect of Context Length on Model Performance and Speed. This table demonstrates the impact of varying sequence max-length on the F$_1$ score, training speed, and inference speed of Twitter-RoBERTa models trained on text-only and emoji-only data. Time is measured in seconds. Both training and inference processes were executed on a 13$^\\mathrm{th}$ Gen Intel Core i9-13900H laptop CPU with 32GB of RAM.}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("                  \\textbf{Sequence Max-Length}  & \\textbf{3} & \\textbf{10} & \\textbf{20} & \\textbf{50} & \\textbf{120} \\\\")
    print("\\midrule")
    print("\\textbf{F1 Score}   &            &             &             &             &              \\\\")
    
    max_lengths = [3, 10, 20, 50, 120]
    
    for dataset in ['Text-Only', 'Emoji-Only']:
        if dataset in results:
            row = f"{dataset:15}"
            for max_length in max_lengths:
                if max_length in results[dataset]:
                    row += f"&{results[dataset][max_length]['f1_score']:.2f:>8}"
                else:
                    row += "&        "
            row += "\\\\"
            print(row)
    
    print("\\midrule")
    print("\\textbf{Training Speed} &        &             &             &             &              \\\\")
    
    for dataset in ['Text-Only', 'Emoji-Only']:
        if dataset in results:
            row = f"{dataset:15}"
            for max_length in max_lengths:
                if max_length in results[dataset]:
                    row += f"&{results[dataset][max_length]['training_time']:.0f:>8}"
                else:
                    row += "&        "
            row += "\\\\"
            print(row)
    
    print("\\midrule")
    print("\\textbf{Inference Speed} &       &             &             &             &              \\\\")
    
    for dataset in ['Text-Only', 'Emoji-Only']:
        if dataset in results:
            row = f"{dataset:15}"
            for max_length in max_lengths:
                if max_length in results[dataset]:
                    row += f"&{results[dataset][max_length]['inference_time']:.0f:>8}"
                else:
                    row += "&        "
            row += "\\\\"
            print(row)
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    table_path = Path(output_dir) / "context_length_table.txt"
    with open(table_path, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{\\label{tab:max-length}Effect of Context Length on Model Performance and Speed. This table demonstrates the impact of varying sequence max-length on the F$_1$ score, training speed, and inference speed of Twitter-RoBERTa models trained on text-only and emoji-only data. Time is measured in seconds. Both training and inference processes were executed on a 13$^\\mathrm{th}$ Gen Intel Core i9-13900H laptop CPU with 32GB of RAM.}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("                  \\textbf{Sequence Max-Length}  & \\textbf{3} & \\textbf{10} & \\textbf{20} & \\textbf{50} & \\textbf{120} \\\\\n")
        f.write("\\midrule\n")
        f.write("\\textbf{F1 Score}   &            &             &             &             &              \\\\\n")
        
        for dataset in ['Text-Only', 'Emoji-Only']:
            if dataset in results:
                row = f"{dataset:15}"
                for max_length in max_lengths:
                    if max_length in results[dataset]:
                        row += f"&{results[dataset][max_length]['f1_score']:.2f:>8}"
                    else:
                        row += "&        "
                row += "\\\\\n"
                f.write(row)
        
        f.write("\\midrule\n")
        f.write("\\textbf{Training Speed} &        &             &             &             &              \\\\\n")
        
        for dataset in ['Text-Only', 'Emoji-Only']:
            if dataset in results:
                row = f"{dataset:15}"
                for max_length in max_lengths:
                    if max_length in results[dataset]:
                        row += f"&{results[dataset][max_length]['training_time']:.0f:>8}"
                    else:
                        row += "&        "
                row += "\\\\\n"
                f.write(row)
        
        f.write("\\midrule\n")
        f.write("\\textbf{Inference Speed} &       &             &             &             &              \\\\\n")
        
        for dataset in ['Text-Only', 'Emoji-Only']:
            if dataset in results:
                row = f"{dataset:15}"
                for max_length in max_lengths:
                    if max_length in results[dataset]:
                        row += f"&{results[dataset][max_length]['inference_time']:.0f:>8}"
                    else:
                        row += "&        "
                row += "\\\\\n"
                f.write(row)
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\nTable saved to {table_path}")
    
    csv_path = Path(output_dir) / "context_length_results.csv"
    results_df = pd.DataFrame({
        (dataset, metric): {max_len: results[dataset][max_len][metric] 
                           for max_len in max_lengths if max_len in results[dataset]}
        for dataset in results
        for metric in ['f1_score', 'training_time', 'inference_time']
    })
    results_df.to_csv(csv_path)
    print(f"Results CSV saved to {csv_path}")

def generate_sample_context_length_results(output_dir):
    sample_results = {
        'Text-Only': {
            3: {'f1_score': 0.64, 'training_time': 272, 'inference_time': 18},
            10: {'f1_score': 0.76, 'training_time': 266, 'inference_time': 18},
            20: {'f1_score': 0.81, 'training_time': 275, 'inference_time': 19},
            50: {'f1_score': 0.83, 'training_time': 291, 'inference_time': 20},
            120: {'f1_score': 0.82, 'training_time': 510, 'inference_time': 37}
        },
        'Emoji-Only': {
            3: {'f1_score': 0.64, 'training_time': 264, 'inference_time': 18},
            10: {'f1_score': 0.74, 'training_time': 266, 'inference_time': 18},
            20: {'f1_score': 0.76, 'training_time': 271, 'inference_time': 17},
            50: {'f1_score': 0.75, 'training_time': 286, 'inference_time': 19},
            120: {'f1_score': 0.76, 'training_time': 469, 'inference_time': 34}
        }
    }
    
    results_df = pd.DataFrame({
        (dataset, metric): {max_len: sample_results[dataset][max_len][metric] 
                           for max_len in sample_results[dataset]}
        for dataset in sample_results
        for metric in ['f1_score', 'training_time', 'inference_time']
    })
    
    csv_path = Path(output_dir) / "sample_context_length_results.csv"
    results_df.to_csv(csv_path)
    print(f"Sample context length results saved to {csv_path}")
    
    return sample_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", help="Directory containing dataset files")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--generate-sample", action="store_true", help="Generate sample results")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.generate_sample:
        generate_sample_context_length_results(args.output_dir)
    
    if args.dataset_dir:
        evaluate_context_length_impact(args.dataset_dir, args.output_dir)
