import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import time

def bootstrap_metrics(y_true, y_pred, B=1000, rng=0):
    """
    Compute bootstrap confidence intervals for macro precision, recall, and F1.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        B: Number of bootstrap samples (default: 1000)
        rng: Random seed (default: 0)
    
    Returns:
        dict: Contains mean and 95% CI for precision, recall, and F1
    """
    np.random.seed(rng)
    n_samples = len(y_true)
    
    # Store bootstrap results
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Bootstrap sampling
    for _ in tqdm(range(B), desc="Bootstrap sampling", leave=False):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Get bootstrap sample
        y_true_boot = [y_true[i] for i in indices]
        y_pred_boot = [y_pred[i] for i in indices]
        
        # Compute macro metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_boot, y_pred_boot, average='macro', zero_division=0
        )
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    # Calculate statistics
    def get_stats(scores):
        return {
            'mean': np.mean(scores),
            'ci_low': np.percentile(scores, 2.5),
            'ci_high': np.percentile(scores, 97.5)
        }
    
    return {
        'precision': get_stats(precision_scores),
        'recall': get_stats(recall_scores),
        'f1': get_stats(f1_scores)
    }

def bootstrap_speed_metrics(train_func, inference_func, B=100, rng=0):
    """
    Compute bootstrap confidence intervals for training and inference times.
    
    Args:
        train_func: Function that returns training time
        inference_func: Function that returns inference time
        B: Number of bootstrap samples (default: 100, lower than metrics due to time cost)
        rng: Random seed (default: 0)
    
    Returns:
        dict: Contains mean and 95% CI for training_time and inference_time
    """
    np.random.seed(rng)
    
    # Store bootstrap results
    training_times = []
    inference_times = []
    
    # Bootstrap sampling
    for _ in tqdm(range(B), desc="Bootstrap speed sampling", leave=False):
        # Training time bootstrap
        train_time = train_func()
        training_times.append(train_time)
        
        # Inference time bootstrap
        inf_time = inference_func()
        inference_times.append(inf_time)
    
    # Calculate statistics
    def get_stats(times):
        return {
            'mean': np.mean(times),
            'ci_low': np.percentile(times, 2.5),
            'ci_high': np.percentile(times, 97.5)
        }
    
    return {
        'training_time': get_stats(training_times),
        'inference_time': get_stats(inference_times)
    }
