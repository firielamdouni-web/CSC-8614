# Exercice 5 : Méthodes de génération avec GPT-2
import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Question 5.a - Configuration
print("Question 5.a - Configuration")
print("-" * 60)
SEED = 42
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
print(f"Prompt: {prompt}")
print(f"Seed: {SEED}")
print()

# Question 5.b - Décodage glouton (greedy)
print("=" * 60)
print("Question 5.b - Décodage glouton")
print("-" * 60)

outputs = model.generate(
    **inputs,
    max_length=50,
)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
print()

# Question 5.c - Sampling avec température, top-k, top-p
print("=" * 60)
print("Question 5.c - Sampling")
print("-" * 60)

def generate_once(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

for s in [1, 2, 3, 4, 5]:
    print(f"SEED {s}")
    print(generate_once(s))
    print("-" * 40)

# Question 5.d - Avec pénalité de répétition
print("\n" + "=" * 60)
print("Question 5.d - Avec pénalité de répétition")
print("-" * 60)

# Sans pénalité
torch.manual_seed(SEED)
out_no_penalty = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
)
text_no_penalty = tokenizer.decode(out_no_penalty[0], skip_special_tokens=True)
print("SANS pénalité:")
print(text_no_penalty)
print()

# Avec pénalité
torch.manual_seed(SEED)
out_penalty = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=2.0,
)
text_penalty = tokenizer.decode(out_penalty[0], skip_special_tokens=True)
print("AVEC pénalité (2.0):")
print(text_penalty)
print()

# Question 5.e - Températures extrêmes
print("=" * 60)
print("Question 5.e - Températures extrêmes")
print("-" * 60)

# Température basse
torch.manual_seed(SEED)
out_low_temp = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    temperature=0.1,
    top_k=50,
    top_p=0.95,
)
text_low_temp = tokenizer.decode(out_low_temp[0], skip_special_tokens=True)
print("Température 0.1:")
print(text_low_temp)
print()

# Température haute
torch.manual_seed(SEED)
out_high_temp = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    temperature=2.0,
    top_k=50,
    top_p=0.95,
)
text_high_temp = tokenizer.decode(out_high_temp[0], skip_special_tokens=True)
print("Température 2.0:")
print(text_high_temp)
print()

# Question 5.f - Beam search
print("=" * 60)
print("Question 5.f - Beam search")
print("-" * 60)

out_beam = model.generate(
    **inputs,
    max_length=50,
    num_beams=5,
    early_stopping=True
)
txt_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
print("Beam search (num_beams=5):")
print(txt_beam)
print()

# Question 5.g - Beam search avec différents nombres de beams
print("=" * 60)
print("Question 5.g - Beam search avec mesure de temps")
print("-" * 60)

for num_beams in [5, 10, 20]:
    start = time.time()
    out = model.generate(
        **inputs,
        max_length=50,
        num_beams=num_beams,
        early_stopping=True
    )
    elapsed = time.time() - start
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"num_beams={num_beams} - Temps: {elapsed:.3f}s")
    print(f"Texte: {txt}")
    print("-" * 40)
