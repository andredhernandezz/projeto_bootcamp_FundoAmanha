# Modelagem de carga alta

## Dependências e análise exploratória

Instale as dependências da modelagem a partir da raiz:

```powershell
python -m pip install -r requirements-modeling.txt
```

O fluxo P1–P8 continua comparando regressão logística, HistGradientBoosting,
Random Forest e XGBoost com features de `dataset_modelagem.csv`. Ele não usa o
holdout. Após escolher uma configuração, a avaliação P9 continua protegida pelo
arquivo `saidas/holdout_usado.lock`:

```powershell
$env:DADOS_DIR=".."
python main.py
python p9_final_holdout.py --confirmo
```

Se `modelo_escolhido.json` ainda apontar para uma seleção CatBoost antiga, rode
novamente `main.py` antes da avaliação P9.

## Modelo de deploy

O deploy usa os Random Forests do pacote `modelagem/CODIGOS_E_RESULTADOS/`, alinhados aos
artefatos e configurações em `visual/assets/`. A geração das features usa os
arquivos originais em `dataset/raw/` e `dataset/interim/`; não usa o CSV antigo
`saidas/dataset_features.csv`, cuja lista de colunas não é compatível.

Treine e salve os modelos locais sem usar validação ou teste:

```powershell
python modelagem/treinar_final.py
```

Para gerar as features RF e executar a previsão:

```powershell
python modelagem/prever.py --saida previsoes.csv
```

Também é possível fornecer um CSV já preparado com as 56/59 features RF:

```powershell
python modelagem/prever.py --entrada caminho\features_rf.csv --saida previsoes.csv
```

Por padrão, os alertas usam os limiares escolhidos na validação (um por cestão).
`--q 0.10` permite explicitamente substituir esses cortes por um corte no
quantil informado. `risco_rf` é a probabilidade bruta do classificador, não uma
probabilidade calibrada.

Os artefatos CatBoost anteriores foram removidos; o deploy usa somente
Random Forest.
