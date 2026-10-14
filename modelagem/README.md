# Modelagem de carga alta
Pré-requisitos: pandas, scikit-learn, xgboost, catboost, matplotlib.
CSVs da EDA na pasta apontada por DADOS_DIR: dataset_modelagem.csv, treino.csv, infos_camadas.csv, materiais_classificados.csv.

1. Desenvolvimento (P1–P8 + figuras; NÃO usa o holdout, ~10 min):
   DADOS_DIR=.. python main.py          # PowerShell: $env:DADOS_DIR=".."; python main.py
2. Avaliação final (UMA vez, após decidir o modelo):
   python p9_final_holdout.py --confirmo
3. Deploy: python treinar_final.py [--com-holdout]   ->  modelo_final/
           python prever.py --entrada saidas/dataset_features.csv --q 0.10 --ultimos 300
Decisões e justificativas: documento "Registro de decisões da modelagem".
