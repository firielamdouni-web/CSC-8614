# Exercice 2 : Tokenizer GPT-2
from transformers import GPT2Tokenizer

# Chargement du tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Question 2.a - Tokenisation
print("Question 2.a")
print("-" * 50)
phrase = "Artificial intelligence is metamorphosing the world!"
tokens = tokenizer.tokenize(phrase)
print("Phrase:", phrase)
print("Tokens:", tokens)
print("Nombre de tokens:", len(tokens))
print()

# Question 2.b - Token IDs
print("Question 2.b")
print("-" * 50)
token_ids = tokenizer.encode(phrase)
print("Token IDs:", token_ids)
print()
print("Détails par token:")
for tid in token_ids:
    txt = tokenizer.decode([tid])
    print(f"  ID {tid}: {repr(txt)}")
print()

# Question 2.c - Observations
print("Question 2.c - Observations")
print("-" * 50)
# Quelques exemples pour observer
print("Mot simple 'world':", tokenizer.tokenize("world"))
print("Mot long 'metamorphosing':", tokenizer.tokenize("metamorphosing"))
print("Avec espace ' is':", tokenizer.tokenize(" is"))
print()

# Question 2.d - Mot très long
print("Question 2.d")
print("-" * 50)
phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."
tokens2 = tokenizer.tokenize(phrase2)
print("Phrase:", phrase2)
print()
print("Tokens:", tokens2)
print()

# Focus sur le mot long
long_word = "antidisestablishmentarianism"
long_word_tokens = tokenizer.tokenize(long_word)
print(f"Tokens pour '{long_word}':")
print(long_word_tokens)
print(f"Nombre de sous-tokens: {len(long_word_tokens)}")
