LangChain Crash Course (70-Day Roadmap)

A practical, end‑to‑end path from Python/ML foundations to production‑grade RAG & agentic systems. This repo is organized for daily progress with weekly mini‑projects, CI/CD, and cloud deployment.

🚀 Goals

Refresh Python + data tooling (NumPy, Pandas, plotting, testing)

Build solid ML fundamentals (scikit‑learn + PyTorch)

Learn modern NLP: embeddings, vector search, transformers

Implement RAG end‑to‑end (ingest → search → synthesize → evaluate)

Explore agent frameworks (LangChain Agents, LlamaIndex agents, multi‑agent patterns)

Ship: FastAPI service, Docker, CI/CD, cloud deploy (GCP/AWS)

🧰 Quick Start (Conda)

# from repo root
conda env create -f environment.yml
conda activate genai
jupyter lab   # or: jupyter notebook

Troubleshooting

Env already exists: conda env update -f environment.yml --prune

"Run conda init": run conda init, restart terminal

CUDA check in notebook

import torch; print('CUDA:', torch.cuda.is_available())

🗂️ Repo Structure

LangChain_CrashCourse/
├─ environment.yml
├─ requirements.txt                # optional (pip users)
├─ README.md
├─ data/                           # sample datasets
├─ notebooks/
│  ├─ 01_python_basics.ipynb
│  ├─ 02_numpy_pandas.ipynb
│  ├─ 03_eda_matplotlib.ipynb
│  ├─ 10_sklearn_pipeline.ipynb
│  ├─ 20_pytorch_tensors.ipynb
│  ├─ 30_embeddings_retrieval.ipynb
│  ├─ 40_rag_end_to_end.ipynb
│  ├─ 50_agents_langchain.ipynb
│  └─ 60_fastapi_docker_ci.ipynb
├─ src/
│  ├─ config.py                    # config helpers
│  ├─ data_utils.py                # IO, cleaning, splitting
│  ├─ retrievers/                  # FAISS/Chroma, BM25, hybrid
│  ├─ rag/                         # chunking, metadata, synthesis
│  ├─ agents/                      # tools, planners, memory
│  └─ api/                         # FastAPI app
├─ tests/                          # pytest
├─ .gitignore
└─ .github/workflows/ci.yml        # GitHub Actions (lint, tests)

Use notebooks/ for learning & experiments; promote stable code to src/ with tests.

📦 Environment Files

environment.yml (Conda): full, reproducible setup (Python, Jupyter, PyTorch, NumPy, Pandas, etc.)

requirements.txt (pip): optional mirror for pip/venv users

# environment.yml (example)
name: genai
dependencies:
  - python=3.11
  - pip
  - pip:
      - torch
      - torchvision
      - torchaudio
      - jupyterlab
      - numpy
      - pandas
      - scikit-learn
      - sentence-transformers
      - faiss-cpu
      - chromadb
      - langchain
      - llama-index
      - fastapi
      - uvicorn
      - pydantic
      - matplotlib
      - mlflow
      - wandb

🧪 Verify Setup (hello.ipynb)

Create/open notebooks/00_hello.ipynb and run:

import torch, numpy as np, pandas as pd
print('Hello from GenAI env!')
print('Torch', torch.__version__, 'CUDA:', torch.cuda.is_available())
print('NumPy OK:', np.ones((2,2)))
print(pd.DataFrame({'Name':['Alice','Bob'],'Score':[85,92]}))

🗺️ 70‑Day Curriculum Map

Use the checklist to track progress. Each bullet maps to one or more notebooks in notebooks/.

Foundations & Data (Weeks 1–3)



Classic ML (Weeks 4–5)



Deep Learning (Weeks 6–7)



NLP, Embeddings & Retrieval (Weeks 8–9)



LLMs, Prompts & RAG (Weeks 10–11)



Agents & Tooling (Week 12)



Shipping & Ops (Week 13)



Capstone (Weeks 14–15)



🧭 How to Work Daily

Pick today’s item from the checklist.

Start a new notebook in notebooks/ with a clear prefix (e.g., 12_feature_engineering.ipynb).

Commit progress daily with a short summary.

Move reusable functions into src/ and add tests.

🧑‍💻 Git Workflow (Solo or Team)

# first setup
git init
git remote add origin <your-repo-url>

# main branch
git checkout -b main

# feature workflow
git checkout -b feature/<short-topic>
# ... make changes ...
git add -A
git commit -m "feat: add retrieval demo over PDFs"
git push -u origin feature/<short-topic>

Open a Pull Request on GitHub → review → merge →

git checkout main
git pull origin main
git branch -d feature/<short-topic>

Common push errors

remote has commits: git pull origin main --allow-unrelated-histories

branch has no upstream: git push -u origin main

wrong branch name (Main vs main): git branch -M main; git push -u origin main

🧾 .gitignore

.venv/
venv/
env/
__pycache__/
.ipynb_checkpoints/
.vscode/
.DS_Store
Thumbs.db

📈 Milestones & Demos

Week 3: EDA report (dataset of choice)

Week 7: Image or text classifier baseline + tracked runs

Week 9: FAQ retriever with FAISS/Chroma

Week 11: Full RAG pipeline + evaluation dashboard

Week 12: Agentic task automator demo

Week 13: FastAPI + Docker + CI → Cloud deploy

Capstone: Production‑ready RAG/Agent with README + demo video

🧠 Tips

Keep notebooks small & focused; re‑run top to bottom before commit.

Pin seeds for reproducibility, save artifacts (models, indices) under /artifacts.

Measure before optimizing; log latency and cost for LLM calls.

Prefer small, composable utilities in src/ + tests.