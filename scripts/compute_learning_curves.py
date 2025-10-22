import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from pathlib import Path
from nltk.tokenize import RegexpTokenizer

def compute_learning_curves(dataset_dir, output_dir):
    datasets = {
        'Text only': f'{dataset_dir}/text_only.csv',
        'Emojis only': f'{dataset_dir}/emoji_only.csv', 
        'Text + Emojis': f'{dataset_dir}/text_emoji.csv'
    }
    
    sample_sizes = [100, 1000, 10000, 100000, 400000]
    results = {}
    
    for name, filepath in datasets.items():
        print(f"\nComputing learning curves for {name}...")
        
        # Load data
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples")
        
        # Clean data
        df = df.dropna(subset=['text', 'label']).copy()
        df['text'] = df['text'].fillna('').astype(str)
        df = df[df['text'].str.strip() != ''].copy()
        
        # Prepare features and labels
        X = df['text']
        y = df['label']
        
        # Use RegexpTokenizer to handle emojis (same as original paper)
        pattern = r'\w+|[^\s]'
        tokenizer = RegexpTokenizer(pattern)
        vectorizer = TfidfVectorizer(max_features=5000, min_df=1, tokenizer=tokenizer.tokenize)
        
        # Fit vectorizer on all data to ensure consistent vocabulary
        X_vectorized = vectorizer.fit_transform(X)
        
        # Split into train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
        
        # Compute learning curves for different sample sizes
        name_results = []
        
        for sample_size in sample_sizes:
            if sample_size > X_train.shape[0]:
                print(f"  Sample size {sample_size} > available training data ({X_train.shape[0]}). Using all training data.")
                actual_size = X_train.shape[0]
            else:
                actual_size = sample_size
                
            print(f"  Training on {actual_size} samples...")
            
            # Subsample training data
            if actual_size < X_train.shape[0]:
                # Stratified sampling to maintain class balance
                indices = np.random.choice(X_train.shape[0], actual_size, replace=False)
                X_train_sample = X_train[indices]
                y_train_sample = y_train.iloc[indices]
            else:
                X_train_sample = X_train
                y_train_sample = y_train
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            start_time = time.time()
            model.fit(X_train_sample, y_train_sample)
            training_time = time.time() - start_time
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"    Accuracy: {accuracy:.3f}, Training time: {training_time:.3f}s")
            
            name_results.append({
                'sample_size': actual_size,
                'accuracy': accuracy,
                'training_time': training_time
            })
        
        results[name] = name_results
    
    # Create the learning curves plot
    create_learning_curves_plot(results, output_dir)
    
    # Save results to CSV
    save_results_to_csv(results, output_dir)
    
    return results

def create_learning_curves_plot(results, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    sample_sizes = [100, 1000, 10000, 100000, 400000]
    
    # Plot each dataset
    colors = {'Text only': 'blue', 'Emojis only': 'green', 'Text + Emojis': 'orange'}
    
    for name, data in results.items():
        sizes = [d['sample_size'] for d in data]
        accuracies = [d['accuracy'] for d in data]
        
        ax.plot(sizes, accuracies, 'o-', color=colors.get(name, 'black'), 
                label=name, linewidth=2, markersize=8)
        
        # Add value labels
        for i, (size, acc) in enumerate(zip(sizes, accuracies)):
            ax.annotate(f'{acc:.3f}', (size, acc), 
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', fontsize=8)
    
    ax.set_xscale('log')
    ax.set_xlabel('Training Sample Size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Learning Curves: Accuracy vs Training Sample Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.50, 0.85)
    
    # Set x-axis ticks to match the sample sizes
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels(sample_sizes)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_path = output_path / "learning_curves_computed.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nLearning curves plot saved to {plot_path}")
    plt.show()

def save_results_to_csv(results, output_dir):
    # Create a combined DataFrame
    all_data = []
    
    for dataset_name, data in results.items():
        for point in data:
            all_data.append({
                'dataset': dataset_name,
                'sample_size': point['sample_size'],
                'accuracy': point['accuracy'],
                'training_time': point['training_time']
            })
    
    df = pd.DataFrame(all_data)
    
    # Save combined results
    csv_path = Path(output_dir) / "learning_curves_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("LEARNING CURVES SUMMARY")
    print("="*80)
    
    # Pivot table for better display
    pivot_df = df.pivot(index='sample_size', columns='dataset', values='accuracy')
    print("\nAccuracy by Sample Size:")
    print(pivot_df.round(3))
    
    print("\nTraining Time by Sample Size (seconds):")
    time_pivot = df.pivot(index='sample_size', columns='dataset', values='training_time')
    print(time_pivot.round(3))
    
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute learning curves from actual data")
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing dataset files")
    parser.add_argument("--output-dir", default="replication/results/plots", help="Output directory")
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    compute_learning_curves(args.dataset_dir, args.output_dir)
