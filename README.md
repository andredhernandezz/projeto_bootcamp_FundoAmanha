# EVCOMX — análise de carga alta

Projeto de análise exploratória e modelagem de eventos de carga alta por cestão. A aplicação Streamlit é mantida como um pacote independente em `visual/`.

## Estrutura

- `dataset/raw/`: arquivos de entrada originais e dicionário de materiais.
- `dataset/interim/`: dados preparados para as etapas seguintes.
- `dataset/processed/`: tabelas analíticas e features.
- `src/`: scripts de preparação e análise.
- `src/modeling/`: engenharia de atributos, análise reproduzível e testes complementares.
- `reports/docs/`, `reports/figures/` e `reports/modeling/`: documentação, gráficos e resultados.
- `visual/`: aplicação Streamlit, modelos, assets e dependências da aplicação.
- `requirements-modeling.txt`: dependências dos scripts de análise.

## Preparar o ambiente de análise

Na raiz do repositório, crie/ative um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Git Bash:
source .venv/Scripts/activate
python -m pip install -r requirements-modeling.txt
```

## Executar o fluxo de análise

Execute os comandos abaixo na raiz do repositório, nesta ordem:

```bash
python src/prepare_dataset.py
python src/eda.py
python src/modeling/build_features.py dataset/raw dataset/interim dataset/processed
python src/modeling/run_analysis.py dataset/raw dataset/processed/features_cestao.csv reports/modeling
python src/modeling/run_supplementary_tests.py
```

`run_analysis.py` executa a análise reproduzível e grava seus artefatos em `reports/modeling/`. A execução pode levar algum tempo.

## Executar a aplicação Streamlit

Instale as dependências específicas do app e inicie a partir da raiz:

```bash
python -m pip install -r visual/requirements.txt
python -m streamlit run visual/app.py
```

Os modelos e arquivos de apoio permanecem em `visual/assets/`. Consulte [visual/README.md](visual/README.md) para detalhes sobre o app, os dados e a interpretação dos resultados.
