# 🔖 Signet

> **Ton compagnon de lecture, sans spoiler.**

Signet est un assistant de lecture personnel propulsé par l'IA. Il connaît ta bibliothèque, tes notes et tes goûts : il te recommande ta prochaine lecture **tous genres confondus** et répond à tes questions sur un livre **sans jamais rien révéler au-delà de la page où tu en es**.

C'est aussi un projet d'apprentissage qui couvre de bout en bout le **Machine Learning classique**, le **RAG** (Retrieval-Augmented Generation), le **fine-tuning de LLM** et la **mise en production**.

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 📚 **Recommandations transversales** | Suggestions basées sur l'« ambiance » des livres (thèmes, rythme, ton) et pas seulement le genre — un thriller et un essai peuvent te plaire pour les mêmes raisons. |
| 🛡️ **Questions sans spoiler** | « Qui est déjà ce personnage ? », « Résume-moi ce qui s'est passé » : Signet ne cherche que dans les pages déjà lues. |
| 📝 **Mémoire de lecture** | Recherche dans tes notes et surlignages (exports Kindle / Kobo) : « Retrouve ce passage sur la mémoire que j'avais noté. » |
| 🗺️ **Carte de lectures** | Visualisation de tes « territoires » de lecteur et des zones encore inexplorées. |
| 🎯 **Défi lecture** | Signet te pousse vers un genre inconnu, mais proche de ce que tu aimes déjà. |

---

## 🧠 Architecture

```
                         ┌──────────────────────┐
   Utilisateur  ───────► │   Agent / Routeur    │
                         │        (LLM)         │
                         └──────────┬───────────┘
            ┌───────────────────────┼────────────────────────┐
            ▼                       ▼                        ▼
 ┌────────────────────┐ ┌───────────────────────┐ ┌────────────────────┐
 │   Recommandeur     │ │  RAG anti-spoiler     │ │   RAG sur notes    │
 │  (ML classique)    │ │ filtre : page < actu. │ │   personnelles     │
 └─────────┬──────────┘ └───────────┬───────────┘ └─────────┬──────────┘
           │                        │                       │
           ▼                        ▼                       ▼
   Dataset Goodreads       Qdrant (vecteurs +       Qdrant (notes &
   + historique perso      métadonnées position)    surlignages)
                                    │
                                    ▼
                    Garde-fou : classifieur de spoilers
                              (modèle fine-tuné)
```

---

## 🛠️ Stack technique

### Socle
- **Python 3.12** · **uv** (gestion des dépendances) · **Git / GitHub** · **Docker**

### Machine Learning — Recommandation
- **Polars** / pandas — manipulation des données
- **scikit-learn** — TF-IDF, clustering, métriques
- **implicit** / **LightFM** — filtrage collaboratif
- **LightGBM** — modèle de ranking
- **UMAP + HDBSCAN** — carte et clustering des lectures

### RAG
- **sentence-transformers** — embeddings multilingues (`multilingual-e5`, `bge-m3`)
- **Qdrant** — base vectorielle avec filtrage par métadonnées (cœur de l'anti-spoiler)
- **BM25 + reranker** (`bge-reranker`) — recherche hybride
- **Ollama** — LLM open source en local · API externe pour comparaison
- **RAGAS** / LLM-as-a-judge — évaluation

### Fine-tuning
- **Hugging Face** — `transformers`, `datasets`, `peft`, `trl`
- **Unsloth** — LoRA / QLoRA rapide sur GPU gratuit (Colab / Kaggle)
- **Petit LLM** (1–8B : Qwen, Llama, Mistral) — style « libraire passionné »
- **CamemBERT / ModernBERT** — classifieur de spoilers
- **Weights & Biases** / MLflow — suivi des expériences

### Application
- **FastAPI** — backend
- **Streamlit** / Gradio — interface (Next.js à terme)
- **SQLite → PostgreSQL** — bibliothèque, progression, notes
- **Langfuse** — observabilité des appels LLM

---

## 📦 Données

| Source | Usage | Licence |
|---|---|---|
| [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads) (Goodreads) | Notes, critiques, critiques annotées « spoiler » | Usage académique |
| [Project Gutenberg](https://www.gutenberg.org) | Textes intégraux pour le RAG | Domaine public |
| [Wikisource](https://fr.wikisource.org) | Classiques français (Hugo, Zola, Dumas, Verne…) | Domaine public |
| Exports Kindle / Kobo | Notes et surlignages personnels | Données perso |

> ⚠️ Seuls des textes du **domaine public** sont indexés intégralement.

---

## 🗺️ Roadmap

- [ ] **Phase 1 — Recommandeur (ML)**
  - [ ] Exploration et nettoyage du dataset Goodreads
  - [ ] Baseline filtrage collaboratif
  - [ ] Embeddings de critiques + modèle hybride
  - [ ] Évaluation : precision@k, NDCG, diversité
  - [ ] Carte des lectures (UMAP)
- [ ] **Phase 2 — RAG anti-spoiler**
  - [ ] Ingestion Gutenberg / Wikisource avec position de chaque chunk
  - [ ] Indexation Qdrant + filtrage par page
  - [ ] Recherche hybride + reranking
  - [ ] Questions globales (résumés hiérarchiques / graphe de personnages)
  - [ ] Jeu d'évaluation fait main (~100 questions) : recall@k, fidélité
- [ ] **Phase 3 — Fine-tuning**
  - [ ] Classifieur de spoilers (F1 sur Goodreads annoté)
  - [ ] Dataset synthétique + LoRA pour le style « libraire »
  - [ ] Comparatif chiffré : base + RAG vs fine-tuné + RAG vs fine-tuné seul
- [ ] **Phase 4 — Application**
  - [ ] Agent routeur
  - [ ] API FastAPI + interface Streamlit
  - [ ] Observabilité (Langfuse) + Docker Compose

---

## 🚀 Démarrage rapide

```bash
# Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Cloner et installer
git clone https://github.com/<ton-pseudo>/signet.git
cd signet
uv sync

# Lancer Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Lancer l'interface
uv run streamlit run src/signet/app.py
```

---

## 📁 Structure du projet

```
signet/
├── data/                 # Données brutes et traitées (non versionnées)
├── notebooks/            # Exploration et prototypes
├── src/signet/
│   ├── recommender/      # Phase 1 — ML
│   ├── rag/              # Phase 2 — ingestion, retrieval, génération
│   ├── finetuning/       # Phase 3 — scripts d'entraînement
│   ├── agent/            # Routeur
│   ├── api/              # FastAPI
│   └── app.py            # Interface Streamlit
├── evals/                # Jeux de test et scripts d'évaluation
├── tests/
├── docker-compose.yml
└── pyproject.toml
```

---

## 📏 Principes du projet

1. **Mesurer avant d'optimiser** — chaque phase a son jeu d'évaluation chiffré.
2. **Comprendre avant d'abstraire** — le premier RAG est écrit à la main, sans framework.
3. **Commencer simple** — on n'ajoute un outil que quand le besoin se fait sentir.

---

## 📄 Licence

MIT
