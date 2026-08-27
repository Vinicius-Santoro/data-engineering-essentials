# Santoro Data Lab — Ciência de Dados 80/20

Base de estudo construída em Material for MkDocs para aprender, praticar e revisar o núcleo de Ciência de Dados.

## Rodando localmente

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash / Windows
pip install -r requirements-docs.txt
mkdocs serve
```

Abra `http://127.0.0.1:8000`.

## Build

```bash
mkdocs build --strict
```

## Publicação no GitHub Pages

O workflow em `.github/workflows/ci.yml` publica automaticamente a branch `main` via `mkdocs gh-deploy`.
