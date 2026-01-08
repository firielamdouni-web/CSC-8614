# Exercice 4 : Probabilités et génération avec GPT-2
import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Chargement du modèle et tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Question 4.a - Probabilités conditionnelles
print("Question 4.a - Probabilités conditionnelles")
print("-" * 60)
phrase = "Artificial intelligence is fascinating."
inputs = tokenizer(phrase, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Conversion en probabilités
probs = torch.softmax(logits, dim=-1)

input_ids = inputs["input_ids"][0]
print(f"Phrase: {phrase}")
print(f"\nProbabilités conditionnelles:")
for t in range(1, len(input_ids)):
    tok_id = input_ids[t].item()
    p = probs[0, t-1, tok_id].item()
    tok_txt = tokenizer.decode([tok_id])
    print(f"Position {t}: {repr(tok_txt):<20} P = {p:.3e}")

# Question 4.b - Log-probabilité et perplexité
print("\n" + "=" * 60)
print("Question 4.b - Log-probabilité et perplexité")
print("-" * 60)

log_probs = torch.log_softmax(logits, dim=-1)
total_logp = 0.0
n = 0

for t in range(1, len(input_ids)):
    tok_id = input_ids[t].item()
    lp = log_probs[0, t-1, tok_id].item()
    total_logp += lp
    n += 1

avg_neg_logp = -total_logp / n
ppl = math.exp(avg_neg_logp)

print(f"Log-probabilité totale: {total_logp:.4f}")
print(f"Log-probabilité moyenne négative: {avg_neg_logp:.4f}")
print(f"Perplexité: {ppl:.4f}")

# Question 4.c - Comparaison phrase correcte vs incorrecte
print("\n" + "=" * 60)
print("Question 4.c - Comparaison phrase correcte vs incorrecte")
print("-" * 60)

phrases = [
    "Artificial intelligence is fascinating.",
    "Artificial fascinating intelligence is."
]

for phrase_test in phrases:
    inputs_test = tokenizer(phrase_test, return_tensors="pt")
    with torch.no_grad():
        outputs_test = model(**inputs_test)
        logits_test = outputs_test.logits
    
    log_probs_test = torch.log_softmax(logits_test, dim=-1)
    input_ids_test = inputs_test["input_ids"][0]
    
    total_logp_test = 0.0
    n_test = 0
    for t in range(1, len(input_ids_test)):
        tok_id = input_ids_test[t].item()
        lp = log_probs_test[0, t-1, tok_id].item()
        total_logp_test += lp
        n_test += 1
    
    avg_neg_logp_test = -total_logp_test / n_test
    ppl_test = math.exp(avg_neg_logp_test)
    
    print(f"\nPhrase: {phrase_test}")
    print(f"  Perplexité: {ppl_test:.4f}")

# Question 4.d - Phrase en français
print("\n" + "=" * 60)
print("Question 4.d - Phrase en français")
print("-" * 60)

phrase_fr = "L'intelligence artificielle est fascinante."
inputs_fr = tokenizer(phrase_fr, return_tensors="pt")

with torch.no_grad():
    outputs_fr = model(**inputs_fr)
    logits_fr = outputs_fr.logits

log_probs_fr = torch.log_softmax(logits_fr, dim=-1)
input_ids_fr = inputs_fr["input_ids"][0]

total_logp_fr = 0.0
n_fr = 0
for t in range(1, len(input_ids_fr)):
    tok_id = input_ids_fr[t].item()
    lp = log_probs_fr[0, t-1, tok_id].item()
    total_logp_fr += lp
    n_fr += 1

avg_neg_logp_fr = -total_logp_fr / n_fr
ppl_fr = math.exp(avg_neg_logp_fr)

print(f"Phrase: {phrase_fr}")
print(f"Perplexité: {ppl_fr:.4f}")

# Question 4.e - Top 10 tokens les plus probables
print("\n" + "=" * 60)
print("Question 4.e - Top 10 tokens suivants")
print("-" * 60)

prefix = "Artificial intelligence is"
inp = tokenizer(prefix, return_tensors="pt")

with torch.no_grad():
    out = model(**inp)
    logits2 = out.logits

last_logits = logits2[0, -1, :]
last_probs = torch.softmax(last_logits, dim=-1)

topk = 10
vals, idx = torch.topk(last_probs, k=topk)

print(f"Préfixe: {prefix}")
print(f"\nTop {topk} tokens les plus probables:")
for i, (p, tid) in enumerate(zip(vals.tolist(), idx.tolist()), 1):
    print(f"{i:2d}. {repr(tokenizer.decode([tid])):<20} P = {p:.3e}")
