# Exercice 3 : Encodages positionnels dans GPT-2
from transformers import GPT2Model
import plotly.express as px
from sklearn.decomposition import PCA

# Question 3.a - Chargement du modèle et extraction des embeddings
print("Question 3.a")
print("-" * 50)
model = GPT2Model.from_pretrained("gpt2")

# Récupération des embeddings positionnels
position_embeddings = model.wpe.weight
print("Shape position embeddings:", position_embeddings.size())
print("n_embd:", model.config.n_embd)
print("n_positions:", model.config.n_positions)
print()

# Question 3.b - Visualisation des 50 premières positions
print("Question 3.b - Visualisation positions 0-50")
print("-" * 50)

# Extraction des 50 premières positions
positions_50 = position_embeddings[:50].detach().cpu().numpy()

# PCA pour réduire à 2 dimensions
pca = PCA(n_components=2)
reduced_50 = pca.fit_transform(positions_50)

# Création du graphique
fig_50 = px.scatter(
    x=reduced_50[:, 0],
    y=reduced_50[:, 1],
    text=[str(i) for i in range(len(reduced_50))],
    color=list(range(len(reduced_50))),
    title="Encodages positionnels GPT-2 (PCA, positions 0-50)",
    labels={"x": "PCA 1", "y": "PCA 2", "color": "Position"}
)
fig_50.update_traces(textposition='top center')
fig_50.write_html("positions_50.html")
print("Graphique sauvegardé dans positions_50.html")
print()

# Question 3.c - Visualisation des 200 premières positions
print("Question 3.c - Visualisation positions 0-200")
print("-" * 50)

# Extraction des 200 premières positions
positions_200 = position_embeddings[:200].detach().cpu().numpy()

# PCA pour réduire à 2 dimensions
pca_200 = PCA(n_components=2)
reduced_200 = pca_200.fit_transform(positions_200)

# Création du graphique
fig_200 = px.scatter(
    x=reduced_200[:, 0],
    y=reduced_200[:, 1],
    text=[str(i) for i in range(len(reduced_200))],
    color=list(range(len(reduced_200))),
    title="Encodages positionnels GPT-2 (PCA, positions 0-200)",
    labels={"x": "PCA 1", "y": "PCA 2", "color": "Position"}
)
fig_200.update_traces(textposition='top center', textfont_size=8)
fig_200.write_html("positions_200.html")
print("Graphique sauvegardé dans positions_200.html")
