# EVCOMX — Streamlit Random Forest (carga alta)

Esta versão usa **somente o Random Forest original** desenvolvido para a análise EVCOMX. Não usa CatBoost.

## Executar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Docker (pipeline containerizado)

Requisitos: Docker (e, opcionalmente, Docker Compose).

```bash
# 1) Construir a imagem (o build roda verificar_modelos.py e falha se as métricas não baterem)
docker build -t evcomx-rf:latest .

# 2) Dashboard / simulador Streamlit em http://localhost:8501
docker run --rm -p 8501:8501 evcomx-rf:latest

# 3) Verificar os modelos
docker run --rm evcomx-rf:latest python pipeline.py verify

# 4) Métricas e business case do teste -> data_out/avaliacao.json
docker run --rm --user root -v "$PWD/data_out:/app/data_out" evcomx-rf:latest \
  python pipeline.py evaluate --output data_out/avaliacao.json

# 5) Previsão em lote (CSV de features em data_in/entrada.csv) -> data_out/previsoes_c1.csv
docker run --rm --user root -v "$PWD/data_in:/app/data_in:ro" -v "$PWD/data_out:/app/data_out" \
  evcomx-rf:latest python pipeline.py predict --cestao 1 --input data_in/entrada.csv --output data_out/previsoes_c1.csv
```

Com Docker Compose:

```bash
docker compose up --build app                          # dashboard
docker compose --profile batch run --rm verify         # verificação
docker compose --profile batch run --rm evaluate       # métricas + business case
CESTAO=2 docker compose --profile batch run --rm predict   # previsão em lote
```

Também há atalhos em `Makefile` (`make build`, `make run`, `make verify`, `make evaluate`, `make predict CESTAO=1`).

### Fluxo do pipeline

`CSV de features (data_in/)` → `pipeline.py predict` (valida colunas, converte vírgula decimal, imputação pela mediana do treino, `predict_proba` do RF) → `risco_rf` + `alerta` pelo limiar do `config.json` (`data_out/`). O cestão 2 exige `carga_alta_c1_ctx` (resultado observado 0/1 do cestão 1). `data_in/entrada.csv` é um exemplo com 20 linhas reais do teste (cestão 1).

**Escopo:** o container executa a **inferência e a avaliação** dos modelos já treinados. O treino e a engenharia de features a partir dos 3 CSVs brutos da empresa (`infos_camadas`, `infos_materiais`, `VariavelResposta`) ficam nos scripts da análise EVCOMX e não fazem parte deste pacote; as features já calculadas entram via CSV.

### Hospedagem no Streamlit Community Cloud

Coloque os arquivos deste pacote em um repositório no GitHub e indique `app.py` como entry point. A pasta `assets/` deve permanecer junto do `app.py`.

### Configuração de caminhos

Se quiser armazenar dados e modelos em diretórios diferentes, configure:

```bash
export EVCOMX_RF_DATA_DIR="/caminho/para/assets"
export EVCOMX_RF_MODEL_DIR="/caminho/para/modelos"
streamlit run app.py
```

Em Windows (PowerShell), use `$env:EVCOMX_RF_DATA_DIR = "..."`.

## Os modelos realmente usados

- `assets/random_forest_c1.joblib`: 180 árvores, 56 variáveis (cestão 1).
- `assets/random_forest_c2.joblib`: 180 árvores, 59 variáveis (cestão 2).
- Pipeline com `SimpleImputer(strategy='median')` e `RandomForestClassifier(n_estimators=180, min_samples_leaf=12, max_features=0.65, random_state=42)`.
- Reajustados **apenas no treino original** e comparados linha a linha com as previsões históricas originais do teste, com desvio máximo da ordem de 1e-16 (erro numérico de ponto flutuante).
- Limiar do cestão 1: `0.24647925398480036`; cestão 2: `0.3528583739753988`. Ambos fixados por F2 na validação, limitando alertas a 35%.
- Não foi feito retreino com validação ou teste. Não há calibração adicional validada do risco.

## Avaliação no teste preservado

| Métrica | Cestão 1 | Cestão 2 |
|---|---:|---:|
| Observações | 180 | 172 |
| Positivos | 26 | 31 |
| Verdadeiros positivos | 22 | 25 |
| Falsos positivos | 44 | 30 |
| Falsos negativos | 4 | 6 |
| Verdadeiros negativos | 110 | 111 |
| Recall | 84,6% | 80,6% |
| Precisão | 33,3% | 45,5% |
| ROC-AUC | 0,8219 | 0,8799 |
| PR-AUC | 0,4104 | 0,7767 |
| Paradas dos VP sinalizados | 59,9 min | 97,1 min |

Os minutos capturados representam tempo de parada **histórico associado aos eventos que teriam recebido alerta**, não tempo comprovadamente recuperado.

## Como interpretar o simulador

O formulário aceita exatamente as features usadas no treino (com nomes e ordem preservados). Começa com um exemplo real do teste, mas o usuário pode alterar os valores. Esses cenários são estatísticos: modificar variáveis independentes pode gerar uma combinação industrial não realizável. O segundo cestão depende do desfecho confirmado do primeiro. No CSV em lote, todas as colunas de features devem existir e não podem ser incluídos alvos ou durações de parada. Valores ausentes dentro dessas colunas são imputados pela mediana aprendida no treino.

## Business case

```text
Ganho líquido (min) = eficácia × minutos dos VP detectados − custo_fp × FP
```

Eficácia 100%, custo_fp 1 minuto e R$ 2.500/min são premissas ilustrativas. Não houve experimento causal que comprove economia. A anualização extrapola linearmente o período do teste, sem considerar sazonalidade.

## Arquivos

`app.py`, `pipeline.py`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `Makefile`, `data_in/`, `data_out/`, `requirements.txt` e pasta `assets/` com 2 modelos joblib, `config.json`, previsões de validação e teste, exemplos e importância por permutação. Um arquivo `verificar_modelos.py` comprova métricas e equivalência.

## Compatibilidade

A última reprodução foi realizada com scikit-learn 1.8.0; o pacote fixa scikit-learn ==1.8.0 para estabilidade de compatibilidade. O formato joblib está sujeito a mudanças entre versões do scikit-learn. Em produção, fixe a versão e rode `verificar_modelos.py` após instalação. Carregue arquivos joblib somente de fontes confiáveis.

### Verificação visual

A lógica do app foi exercitada por testes automatizados que percorrem as seis funções de páginas com controles simulados. O Streamlit não estava disponível no ambiente de criação, portanto a interface em navegador real ainda deve ser validada após a instalação local.

## Correção visual (novembro de 2026)

Esta edição resolve o problema de textos e KPIs quase brancos sobre fundo branco
no simulador, observado quando o Streamlit inicia com tema escuro. O arquivo
`.streamlit/config.toml` força o tema claro com a paleta EVCOMX. O `app.py` também
contém regras CSS explícitas para o contraste de rótulos, campos, cartões de métricas,
barra lateral, expansão das variáveis avançadas e alertas. Nenhum modelo, dado,
limiar ou cálculo foi alterado em relação à versão Random Forest original.

**Importante:** copie também a pasta oculta `.streamlit/`, não apenas o `app.py`.
No Streamlit Community Cloud, faça commit dessa pasta e reinicie o aplicativo.
Se o tema antigo persistir no navegador, recarregue forçadamente (Ctrl + Shift + R).
