import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from nltk.tokenize import RegexpTokenizer
from bootstrap_utils import bootstrap_metrics, bootstrap_speed_metrics

def evaluate_all_lr_models(dataset_dir, output_dir):
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
        
        train_ids = pd.read_csv(f'{dataset_dir}/splits/train_ids.txt', header=None)[0].tolist()
        test_ids = pd.read_csv(f'{dataset_dir}/splits/test_ids.txt', header=None)[0].tolist()
        
        train_df = df[df['post_id'].isin(train_ids)]
        test_df = df[df['post_id'].isin(test_ids)]
        
        # Determine the text column name (could be 'text' or 'emojis')
        text_col = 'emojis' if 'emojis' in df.columns else 'text'
        
        # Clean text data - remove NaN values and empty strings
        train_df = train_df.dropna(subset=[text_col]).copy()
        test_df = test_df.dropna(subset=[text_col]).copy()
        train_df[text_col] = train_df[text_col].fillna('').astype(str)
        test_df[text_col] = test_df[text_col].fillna('').astype(str)
        
        # Remove empty texts
        train_df = train_df[train_df[text_col].str.strip() != ''].copy()
        test_df = test_df[test_df[text_col].str.strip() != ''].copy()
        
        # Use RegexpTokenizer to handle emojis (same as original paper)
        pattern = r'\w+|[^\s]'
        tokenizer = RegexpTokenizer(pattern)
        vectorizer = TfidfVectorizer(max_features=5000, min_df=1, tokenizer=tokenizer.tokenize)
        
        X_train = vectorizer.fit_transform(train_df[text_col])
        X_test = vectorizer.transform(test_df[text_col])
        y_train = train_df['label']
        y_test = test_df['label']
        
        # Single training and inference for initial metrics
        start_time = time.time()
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=['bearish', 'bullish'])
        
        # Compute bootstrap confidence intervals for accuracy metrics
        print(f"  Computing bootstrap confidence intervals for accuracy metrics...")
        bootstrap_results = bootstrap_metrics(y_test.tolist(), y_pred.tolist(), B=1000, rng=42)
        
        # Compute bootstrap confidence intervals for speed metrics
        print(f"  Computing bootstrap confidence intervals for speed metrics...")
        
        def train_func():
            start_time = time.time()
            temp_model = LogisticRegression(random_state=np.random.randint(0, 10000), max_iter=1000)
            temp_model.fit(X_train, y_train)
            return time.time() - start_time
        
        def inference_func():
            start_time = time.time()
            temp_pred = model.predict(X_test)
            return time.time() - start_time
        
        speed_bootstrap_results = bootstrap_speed_metrics(train_func, inference_func, B=50, rng=42)
        
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
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    create_performance_table(results, output_dir)
    create_confusion_matrices(confusion_matrices, output_dir)
    
    return results, confusion_matrices

def create_performance_table(results, output_dir):
    print("\n" + "="*100)
    print("LOGISTIC REGRESSION PERFORMANCE TABLE WITH BOOTSTRAP CONFIDENCE INTERVALS")
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
            row += f"{mean_val:>6.3f} [{ci_low:>5.3f}, {ci_high:>5.3f}]"
        print(row)
    
    print("="*100)

def create_confusion_matrices(confusion_matrices, output_dir):
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
    
    plt.suptitle('Confusion Matrices of Logistic Regression Models', 
                 y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = Path(output_dir) / "lr_confusion_matrices.png"
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
    evaluate_all_lr_models(args.dataset_dir, args.output_dir)
