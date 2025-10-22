import pandas as pd
import numpy as np
import re
import time
from collections import Counter
from scipy.stats import chi2_contingency, norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
import os

def get_num_cores():
    """Get number of cores minus 2 for parallel processing"""
    return max(1, mp.cpu_count() - 2)

def get_emoji_alias(emoji):
    try:
        import unicodedata
        return unicodedata.name(emoji, '').replace(' ', '_').lower()
    except Exception:
        return ''

def extract_unique_emojis(s):
    if not isinstance(s, str):
        return set()
    pattern = re.compile(r'[\U0001F1E6-\U0001F1FF]|[\U0001F300-\U0001F5FF]|[\U0001F600-\U0001F64F]|[\U0001F680-\U0001F6FF]|[\U0001F700-\U0001F77F]|[\U0001F780-\U0001F7FF]|[\U0001F800-\U0001F8FF]|[\U0001F900-\U0001F9FF]|[\U0001FA00-\U0001FAFF]|[\U00002600-\U000026FF]|[\U00002700-\U000027BF]')
    return set(pattern.findall(s))

def count_presence_chunk(args):
    """Helper function for parallel emoji counting"""
    chunk, col = args
    cnt = Counter()
    for x in chunk[col].values:
        ems = extract_unique_emojis(x)
        for e in ems:
            cnt[e] += 1
    return cnt

def count_presence(df, col, desc="Counting emojis"):
    """Count emojis in parallel using multiple cores with progress tracking"""
    num_cores = get_num_cores()
    
    # Split dataframe into chunks
    chunk_size = len(df) // num_cores
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Prepare arguments for parallel processing
    args_list = [(chunk, col) for chunk in chunks]
    
    # Process in parallel with progress tracking
    cnt = Counter()
    with Pool(processes=num_cores) as pool:
        with tqdm(total=len(chunks), desc=desc) as pbar:
            for result in pool.imap_unordered(count_presence_chunk, args_list):
                cnt.update(result)
                pbar.update(1)
    
    return cnt

def build_contingency(cnt1, n1, cnt2, n2):
    emojis = sorted(set(cnt1.keys()) | set(cnt2.keys()))
    a = np.array([cnt1.get(e,0) for e in emojis], dtype=int)
    b = np.array([cnt2.get(e,0) for e in emojis], dtype=int)
    return emojis, np.vstack([a, b]), n1, n2

def cramers_v_from_chi2(chi2, n):
    return np.sqrt(chi2 / n)

def js_divergence(p, q):
    m = 0.5*(p+q)
    p = np.where(p==0, 1e-12, p)
    q = np.where(q==0, 1e-12, q)
    m = np.where(m==0, 1e-12, m)
    kl_pm = np.sum(p*np.log(p/m))
    kl_qm = np.sum(q*np.log(q/m))
    return 0.5*(kl_pm+kl_qm)

def per_emoji_log_odds(M, n1, n2, alpha=0.5):
    a = M[0].astype(float)
    b = M[1].astype(float)
    A = a + alpha
    B = b + alpha
    F1 = (n1 - a) + alpha
    F2 = (n2 - b) + alpha
    lo1 = np.log(A) - np.log(F1)
    lo2 = np.log(B) - np.log(F2)
    d = lo1 - lo2
    var = 1.0/A + 1.0/F1 + 1.0/B + 1.0/F2
    se = np.sqrt(var)
    z = d / se
    p = 2*(1 - norm.cdf(np.abs(z)))
    ci_lo = d - 1.96*se
    ci_hi = d + 1.96*se
    return d, ci_lo, ci_hi, p

def run_single_experiment(stock_df, tweets_df, experiment_num, random_state):
    """Run a single experiment with one random sample"""
    print(f"\n--- Experiment {experiment_num + 1} (random_state={random_state}) ---")
    
    # Sample tweets dataset
    tw_sample = tweets_df.sample(n=len(stock_df), random_state=random_state)
    print(f"Sampled {len(tw_sample)} tweets")
    
    # Count emojis
    print("Counting emojis in stock dataset...")
    c1 = count_presence(stock_df, 'emojis', f"Stock dataset - Exp {experiment_num + 1}")
    print("Counting emojis in tweets dataset...")
    c2 = count_presence(tw_sample, 'emojis', f"Tweets dataset - Exp {experiment_num + 1}")
    
    # Build contingency table
    ems, M, n1, n2 = build_contingency(c1, len(stock_df), c2, len(tw_sample))
    
    # Calculate main statistics
    chi2, p_chi, _, _ = chi2_contingency(M, correction=False)
    V = cramers_v_from_chi2(chi2, M.sum())
    
    a = M[0].astype(float)
    b = M[1].astype(float)
    p1 = a / a.sum() if a.sum() > 0 else np.zeros_like(a, dtype=float)
    p2 = b / b.sum() if b.sum() > 0 else np.zeros_like(b, dtype=float)
    JSD = js_divergence(p1, p2)
    
    # Calculate per-emoji log odds
    d, lo, hi, pvals = per_emoji_log_odds(M, n1, n2, alpha=0.5)
    
    # Count significant emojis (p < 0.05)
    significant_emojis = np.sum(pvals < 0.05)
    
    # Count emojis with significant log odds (CI doesn't include 0)
    significant_log_odds = np.sum((lo > 0) | (hi < 0))
    
    results = {
        'experiment': experiment_num + 1,
        'random_state': random_state,
        'n_emojis': len(ems),
        'chi2': chi2,
        'p_chi': p_chi,
        'cramers_v': V,
        'jsd': JSD,
        'significant_emojis': significant_emojis,
        'significant_log_odds': significant_log_odds,
        'mean_log_odds': np.mean(d),
        'std_log_odds': np.std(d),
        'max_log_odds': np.max(d),
        'min_log_odds': np.min(d)
    }
    
    print(f"Results: E={len(ems)}, chi2={chi2:.3f}, p={p_chi:.3g}, CramerV={V:.3f}, JSD={JSD:.3f}")
    print(f"Significant emojis: {significant_emojis}, Significant log odds: {significant_log_odds}")
    
    return results

def calculate_percentiles(results_list, metric):
    """Calculate percentiles for a given metric"""
    values = [r[metric] for r in results_list]
    return {
        'min': np.min(values),
        'max': np.max(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'p5': np.percentile(values, 5),
        'p95': np.percentile(values, 95),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75)
    }

# Set multiprocessing start method for Windows compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print(f"Using {get_num_cores()} cores for parallel processing")
    
    # Load datasets
    print("Loading stocktwits dataset...")
    stock = pd.read_csv('emojis_all.csv')
    print(f"Stock dataset has {len(stock)} rows")
    
    print("Loading tweets dataset...")
    tw_full = pd.read_csv('tweets_emojis.csv')
    print(f"Tweets dataset has {len(tw_full)} rows")
    
    # Run experiments
    n_experiments = 10
    results_list = []
    
    print(f"\nRunning {n_experiments} experiments with different random samples...")
    
    for i in range(n_experiments):
        random_state = 42 + i  # Different random state for each experiment
        results = run_single_experiment(stock, tw_full, i, random_state)
        results_list.append(results)
    
    # Calculate summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS ACROSS EXPERIMENTS")
    print("="*60)
    
    metrics = ['n_emojis', 'chi2', 'p_chi', 'cramers_v', 'jsd', 'significant_emojis', 
               'significant_log_odds', 'mean_log_odds', 'std_log_odds', 'max_log_odds', 'min_log_odds']
    
    summary_stats = {}
    for metric in metrics:
        summary_stats[metric] = calculate_percentiles(results_list, metric)
    
    # Print results
    for metric in metrics:
        stats = summary_stats[metric]
        print(f"\n{metric.upper()}:")
        print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
        print(f"  5th percentile: {stats['p5']:.4f}, 95th percentile: {stats['p95']:.4f}")
        print(f"  25th percentile: {stats['p25']:.4f}, 75th percentile: {stats['p75']:.4f}")
    
    # Save detailed results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('sampling_experiment_results.csv', index=False)
    print(f"\nDetailed results saved to 'sampling_experiment_results.csv'")
    
    # Save summary statistics
    summary_df = pd.DataFrame(summary_stats).T
    summary_df.to_csv('sampling_experiment_summary.csv')
    print(f"Summary statistics saved to 'sampling_experiment_summary.csv'")
    
    print("\nExperiment completed!")
