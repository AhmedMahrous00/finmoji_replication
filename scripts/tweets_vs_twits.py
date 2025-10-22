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

def count_presence_sequential(df, col):
    """Sequential emoji counting for use within parallel processes"""
    cnt = Counter()
    for x in df[col].values:
        ems = extract_unique_emojis(x)
        for e in ems:
            cnt[e] += 1
    return cnt

def bootstrap_iteration(args):
    """Helper function for parallel bootstrap iterations"""
    df1, df2, col, iteration, n1, n2, random_state = args
    rng = np.random.default_rng(random_state + iteration)
    i1 = rng.integers(0, n1, n1)
    i2 = rng.integers(0, n2, n2)
    b1 = df1.iloc[i1]
    b2 = df2.iloc[i2]
    # Use sequential counting to avoid nested multiprocessing
    c1 = count_presence_sequential(b1, col)
    c2 = count_presence_sequential(b2, col)
    ems, M, _, _ = build_contingency(c1, len(b1), c2, len(b2))
    chi2, p, _, _ = chi2_contingency(M, correction=False)
    v = cramers_v_from_chi2(chi2, M.sum())
    a = M[0].astype(float)
    b = M[1].astype(float)
    p1 = a / a.sum() if a.sum()>0 else np.zeros_like(a, dtype=float)
    p2 = b / b.sum() if b.sum()>0 else np.zeros_like(b, dtype=float)
    jsd = js_divergence(p1, p2)
    return v, jsd

def bootstrap_stats(df1, df2, col, B=500, random_state=0):
    """Parallel bootstrap statistics computation with progress tracking"""
    num_cores = get_num_cores()
    n1, n2 = len(df1), len(df2)
    
    # Prepare arguments for parallel processing
    args_list = [(df1, df2, col, i, n1, n2, random_state) for i in range(B)]
    
    # Process in parallel with progress tracking
    vs, jsds = [], []
    with Pool(processes=num_cores) as pool:
        with tqdm(total=B, desc="Bootstrap sampling") as pbar:
            for result in pool.imap_unordered(bootstrap_iteration, args_list):
                v, jsd = result
                vs.append(v)
                jsds.append(jsd)
                pbar.update(1)
    
    # Calculate percentiles
    v_lo, v_hi = np.percentile(vs, [2.5, 97.5])
    j_lo, j_hi = np.percentile(jsds, [2.5, 97.5])
    return (float(np.mean(vs)), float(v_lo), float(v_hi)), (float(np.mean(jsds)), float(j_lo), float(j_hi))

def permutation_iteration(args):
    """Helper function for parallel permutation test iterations"""
    all_emojis, combined_counts, cnt1_sum, iteration, random_state = args
    rng = np.random.default_rng(random_state + iteration)
    
    # Shuffle the combined emoji labels
    shuffled = rng.permutation(combined_counts)
    
    # Split back into two groups
    perm_cnt1 = Counter(shuffled[:cnt1_sum])
    perm_cnt2 = Counter(shuffled[cnt1_sum:])
    
    # Compute JSD for permuted data
    a_perm = np.array([perm_cnt1.get(e, 0) for e in all_emojis], dtype=float)
    b_perm = np.array([perm_cnt2.get(e, 0) for e in all_emojis], dtype=float)
    p1_perm = a_perm / a_perm.sum() if a_perm.sum() > 0 else np.zeros_like(a_perm, dtype=float)
    p2_perm = b_perm / b_perm.sum() if b_perm.sum() > 0 else np.zeros_like(b_perm, dtype=float)
    perm_jsd = js_divergence(p1_perm, p2_perm)
    
    return perm_jsd

def permutation_test_jsd(cnt1, cnt2, n1, n2, B=1000, random_state=0):
    """Parallel permutation test for JSD with progress tracking"""
    num_cores = get_num_cores()
    
    # Get all unique emojis
    all_emojis = sorted(set(cnt1.keys()) | set(cnt2.keys()))
    
    # Create combined counts
    combined_counts = []
    for emoji in all_emojis:
        combined_counts.extend([emoji] * cnt1.get(emoji, 0))
        combined_counts.extend([emoji] * cnt2.get(emoji, 0))
    
    # Original JSD
    a = np.array([cnt1.get(e, 0) for e in all_emojis], dtype=float)
    b = np.array([cnt2.get(e, 0) for e in all_emojis], dtype=float)
    p1 = a / a.sum() if a.sum() > 0 else np.zeros_like(a, dtype=float)
    p2 = b / b.sum() if b.sum() > 0 else np.zeros_like(b, dtype=float)
    original_jsd = js_divergence(p1, p2)
    
    # Prepare arguments for parallel processing
    cnt1_sum = sum(cnt1.values())
    args_list = [(all_emojis, combined_counts, cnt1_sum, i, random_state) for i in range(B)]
    
    # Process in parallel with progress tracking
    perm_jsds = []
    with Pool(processes=num_cores) as pool:
        with tqdm(total=B, desc="Permutation test") as pbar:
            for result in pool.imap_unordered(permutation_iteration, args_list):
                perm_jsds.append(result)
                pbar.update(1)
    
    # Calculate p-value
    p_value = np.mean(np.array(perm_jsds) >= original_jsd)
    
    return original_jsd, p_value, perm_jsds

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

# Set multiprocessing start method for Windows compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    
    print(f"Using {get_num_cores()} cores for parallel processing")
    
    print("Loading stocktwits dataset...")
    stock = pd.read_csv('emojis_all.csv')
    print(f"Stock dataset has {len(stock)} rows")

    print("Loading tweets dataset...")
    tw_full = pd.read_csv('tweets_emojis.csv')
    print(f"Tweets dataset has {len(tw_full)} rows")

    print(f"Randomly sampling {len(stock)} rows from tweets dataset...")
    tw = tw_full.sample(n=len(stock), random_state=42)
    print(f"Sampled tweets dataset has {len(tw)} rows")

    print("Counting emojis in stock dataset...")
    c1 = count_presence(stock, 'emojis', "Stock dataset emoji counting")
    print("Counting emojis in tweets dataset...")
    c2 = count_presence(tw, 'emojis', "Tweets dataset emoji counting")
    ems, M, n1, n2 = build_contingency(c1, len(stock), c2, len(tw))

    chi2, p_chi, _, _ = chi2_contingency(M, correction=False)
    V = cramers_v_from_chi2(chi2, M.sum())

    a = M[0].astype(float); b = M[1].astype(float)
    p1 = a / a.sum() if a.sum()>0 else np.zeros_like(a, dtype=float)
    p2 = b / b.sum() if b.sum()>0 else np.zeros_like(b, dtype=float)
    JSD = js_divergence(p1, p2)

    print("Starting bootstrap analysis...")
    (V_mean, V_lo, V_hi), (J_mean, J_lo, J_hi) = bootstrap_stats(stock, tw, 'emojis', B=300, random_state=42)

    print("Starting permutation test for JSD...")
    original_jsd, jsd_p_value, perm_jsds = permutation_test_jsd(c1, c2, n1, n2, B=1000, random_state=42)

    print("Computing per-emoji log odds...")
    d, lo, hi, pvals = per_emoji_log_odds(M, n1, n2, alpha=0.5)

    stock_counts = np.array([c1.get(e, 0) for e in ems])
    twitter_counts = np.array([c2.get(e, 0) for e in ems])

    df_out = pd.DataFrame({
        'emoji': ems,
        'emoji_alias': [get_emoji_alias(e) for e in ems],
        'stocktwits_count': stock_counts,
        'twitter_count': twitter_counts,
        'log_odds': d,
        'lower_CI': lo,
        'upper_CI': hi,
        'p_value': pvals
    })
    print("Saving results to CSV...")
    df_out.sort_values('p_value').to_csv('emoji_log_odds.csv', index=False)

    print("Creating visualization...")
    top = df_out.reindex(df_out['log_odds'].abs().sort_values(ascending=False).head(20).index).copy()
    top = top.sort_values('log_odds')
    plt.figure(figsize=(8,6))
    plt.barh(top['emoji'], top['log_odds'])
    plt.axvline(0, linestyle='--')
    plt.tight_layout()
    plt.savefig('top20_log_odds.png', dpi=200)
    print("Visualization saved as top20_log_odds.png")

    E = len(ems)
    print(f'E={E}')
    print(f'chi2={chi2:.3f}, p={p_chi:.3g}, CramerV={V:.3f}, boot_CI=[{V_lo:.3f},{V_hi:.3f}]')
    print(f'JSD={JSD:.3f}, boot_CI=[{J_lo:.3f},{J_hi:.3f}], perm_p={jsd_p_value:.3g}')
    
    # Save summary metrics to CSV
    summary_metrics = pd.DataFrame({
        'metric': ['chi2', 'p_chi', 'cramers_v', 'cramers_v_lo', 'cramers_v_hi', 
                   'jsd', 'jsd_lo', 'jsd_hi', 'jsd_perm_p_value', 
                   'n_emojis', 'n_stock_samples', 'n_twitter_samples', 'n_bootstrap', 'n_permutations'],
        'value': [chi2, p_chi, V, V_lo, V_hi, 
                  JSD, J_lo, J_hi, jsd_p_value, 
                  E, n1, n2, 300, 1000]
    })
    summary_metrics.to_csv('summary_metrics.csv', index=False)
    print("Summary metrics saved to 'summary_metrics.csv'")
