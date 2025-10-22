import argparse
import csv
import os
import re
from transformers import AutoTokenizer

EMOJI_RE = re.compile(
    "[" 
    "\U0001F300-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U00002B00-\U00002BFF"
    "\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF"
    "]"
)

DEFAULT_POSTS = [
    "BTC to the moon 🚀🚀🚀! New ATH soon? 📈",
    "Bears in control… 😞📉 but maybe a bounce 🤔",
    "Massive volume! 🔥🔥 Exchange outflows rising 🏦➡️🔒",
    "Paper hands 😬 sold too early… diamond hands 💎🙌 stay calm",
    "ETH upgrades incoming 🛠️🚀 devs shipping fast ⚡",
    "LOL this dump 😂😂 then instant pump 🚀",
]

DEFAULT_MODELS = {
    "Twitter-RoBERTa": "cardiffnlp/twitter-roberta-base",
    "FinBERT": "ProsusAI/finbert",
    "DistilBERT": "distilbert-base-uncased",
    "FinancialBERT": "yiyanghkust/finbert-pretrain",
    "CryptoBERT": "ahmedkhan/crypto-bert-base-uncased",
    "FinTwitBERT": "amphora/FinTwit-BERT-base-uncased",
}

def is_emoji_char(ch):
    return EMOJI_RE.match(ch) is not None

def emoji_char_count(s):
    return sum(1 for ch in s if is_emoji_char(ch))

def tokenize_with_offsets(tokenizer, text):
    out = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=False,
    )
    ids = out["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    offsets = out["offset_mapping"]
    return toks, offsets

def preserved_emojis(tokenizer, text, tokens, offsets):
    preserved = 0
    for (s, e) in offsets:
        if e - s == 1 and is_emoji_char(text[s:e]):
            preserved += 1
    return preserved

def run(models, posts, outdir):
    os.makedirs(outdir, exist_ok=True)
    per_post_path = os.path.join(outdir, "tokenizer_audit_per_post.csv")
    summary_path = os.path.join(outdir, "tokenizer_audit_summary.csv")
    with open(per_post_path, "w", newline="", encoding="utf-8") as f_post, \
         open(summary_path, "w", newline="", encoding="utf-8") as f_sum:
        wp = csv.writer(f_post)
        ws = csv.writer(f_sum)
        wp.writerow(["model","post_id","text","tokens","emoji_count","preserved_emojis","preserved_rate","unk_tokens"])
        ws.writerow(["model","total_emoji","total_preserved","preserved_rate","total_unk_tokens"])
        for name, hf_id in models.items():
            try:
                tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
            except Exception as e:
                continue
            total_emoji = 0
            total_pres = 0
            total_unk = 0
            print(f"\n=== {name} ({hf_id}) ===")
            for i, text in enumerate(posts):
                toks, offs = tokenize_with_offsets(tok, text)
                unk = sum(1 for t in toks if t == tok.unk_token)
                ec = emoji_char_count(text)
                pres = preserved_emojis(tok, text, toks, offs)
                total_emoji += ec
                total_pres += pres
                total_unk += unk
                rate = (pres / ec) if ec > 0 else 0.0
                wp.writerow([name, i, text, " ".join(toks), ec, pres, f"{rate:.4f}", unk])
                print(f"[post {i}]")
                print(text)
                print("TOKENS:", toks)
                print(f"emoji_count={ec} preserved={pres} rate={rate:.4f} unk_tokens={unk}\n")
            model_rate = (total_pres / total_emoji) if total_emoji > 0 else 0.0
            ws.writerow([name, total_emoji, total_pres, f"{model_rate:.4f}", total_unk])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="results/tokenizer_audit")
    p.add_argument("--models", type=str, nargs="*", default=[])
    p.add_argument("--posts", type=str, nargs="*", default=[])
    args = p.parse_args()
    models = DEFAULT_MODELS.copy()
    if args.models:
        models = {}
        for spec in args.models:
            k, v = spec.split("=", 1)
            models[k] = v
    posts = DEFAULT_POSTS if not args.posts else args.posts
    run(models, posts, args.outdir)
