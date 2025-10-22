import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
import time
from pathlib import Path

def evaluate_all_baseline_models(dataset_dir, output_dir):
    print("Loading emoji-only dataset...")
    df = pd.read_csv(f'{dataset_dir}/emoji_only.csv')
    
    # Clean text data - remove NaN values and empty strings
    df = df.dropna(subset=['text', 'label']).copy()
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip() != ''].copy()
    
    # Use 80% train, 20% test split (as specified in the paper)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Use RegexpTokenizer to handle emojis (same as original paper)
    pattern = r'\w+|[^\s]'
    tokenizer = RegexpTokenizer(pattern)
    vectorizer = TfidfVectorizer(max_features=5000, min_df=1, tokenizer=tokenizer.tokenize)
    
    X_train = vectorizer.fit_transform(train_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    y_train = train_df['label']
    y_test = test_df['label']
    
    results = {}
    
    models = {
        'MNB': MultinomialNB(),
        'RF': RandomForestClassifier(random_state=42, n_jobs=-1),  # Use all CPU cores as specified
        'LGBM': LGBMClassifier(random_state=42, verbose=-1),
        'SVC': LinearSVC(random_state=42),
        'Reg': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'recall': report['macro avg']['recall'],
            'precision': report['macro avg']['precision'],
            'f1': report['macro avg']['f1-score'],
            'training_time': training_time,
            'inference_time': inference_time
        }
    
    create_baseline_comparison_table(results, output_dir)
    
    return results

def create_baseline_comparison_table(results, output_dir):
    print("\n" + "="*90)
    print("BASELINE MODELS COMPARISON TABLE")
    print("="*90)
    
    print(f"{'Metric':<15} {'MNB':>8} {'RF':>8} {'LGBM':>8} {'SVC':>8} {'Reg':>8} {'Tr':>8}")
    print("-" * 90)
    
    print("\nAccuracy Metrics:")
    metrics = ['recall', 'precision', 'f1']
    metric_names = ['Recall', 'Precision', 'F1 Score']
    
    for metric, name in zip(metrics, metric_names):
        row = f"{name:<15}"
        for model in ['MNB', 'RF', 'LGBM', 'SVC', 'Reg']:
            if model in results:
                row += f"{results[model][metric]:>8.2f}"
            else:
                row += f"{'N/A':>8}"
        row += f"{'N/A':>8}"  # Transformer results would be from separate evaluation
        print(row)
    
    print("\nSpeed Metrics (seconds):")
    speed_metrics = ['training_time', 'inference_time']
    speed_names = ['Training time', 'Inference time']
    
    for metric, name in zip(speed_metrics, speed_names):
        row = f"{name:<15}"
        for model in ['MNB', 'RF', 'LGBM', 'SVC', 'Reg']:
            if model in results:
                row += f"{results[model][metric]:>8.3f}"
            else:
                row += f"{'N/A':>8}"
        row += f"{'N/A':>8}"  # Transformer results would be from separate evaluation
        print(row)
    
    print("="*90)
    
    table_path = Path(output_dir) / "baseline_models_comparison_table.txt"
    with open(table_path, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{\\label{table:other_models}Performance Comparison of Baseline Machine Learning Models on Emoji Data. This table evaluates the accuracy and speed of various baseline models, including Multinomial Na\"{i}ve Bayes (MNB), Random Forest (RF), LightGBM (LGBM), and LinearSVC (SVC), in comparison to the Logistic Regression (Reg) and Transformer (Tr) models, all using emoji-only data. Time is measured in seconds. Both training and inference processes were executed on a 13$^\\mathrm{th}$ Gen Intel Core i9-13900H laptop CPU with 32GB of RAM.}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("\\toprule\n")
        f.write("                    & \\textbf{MNB} & \\textbf{RF} & \\textbf{LGBM} & \\textbf{SVC} & \\textbf{Reg} & \\textbf{Tr}\\\\\n")
        f.write("\\midrule\n")
        f.write("\\textbf{Accuracy} &&&&&&\\\\\n")
        
        for metric, name in zip(metrics, metric_names):
            row = f"{name:15}"
            for model in ['MNB', 'RF', 'LGBM', 'SVC', 'Reg']:
                if model in results:
                    row += f"&{results[model][metric]:>8.2f}"
                else:
                    row += "&        "
            row += "&        "
            row += "\\\\\n"
            f.write(row)
        
        f.write("\\midrule\n")
        f.write("\\textbf{Speed} &                   &        &              &&&                        \\\\\n")
        
        for metric, name in zip(speed_metrics, speed_names):
            row = f"{name:15}"
            for model in ['MNB', 'RF', 'LGBM', 'SVC', 'Reg']:
                if model in results:
                    row += f"&{results[model][metric]:>8.3f}"
                else:
                    row += "&        "
            row += "&        "
            row += "\\\\\n"
            f.write(row)
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\nTable saved to {table_path}")
    
    csv_path = Path(output_dir) / "baseline_models_results.csv"
    results_df = pd.DataFrame(results).T
    results_df.to_csv(csv_path)
    print(f"Results CSV saved to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing dataset files")
    parser.add_argument("--output-dir", default="replication/results/plots", help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate_all_baseline_models(args.dataset_dir, args.output_dir)
