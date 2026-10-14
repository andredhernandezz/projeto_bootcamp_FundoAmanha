from catboost import CatBoostClassifier
import pandas as pd
import os

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))

caminho_modelo1 = os.path.join(
    DIRETORIO_ATUAL, "modelagem", "modelo_final",
    "modelo_cestao1.cbm"
)
caminho_modelo2 = os.path.join(
    DIRETORIO_ATUAL, "modelagem", "modelo_final",
    "modelo_cestao2.cbm"
)
caminho_dados = os.path.join(
    DIRETORIO_ATUAL, "modelagem", "saidas",
    "dataset_features.csv"
)

caminho_saida = os.path.join(
    DIRETORIO_ATUAL, "modelagem", "saidas",
    "resultado_previsoes.csv"
)

# Classe que representa carga alta no treinamento.
CLASSE_CARGA_ALTA = 1

# Limiar provisório: configurar conforme o critério escolhido.
# 0.364 foi uma das alternativas da tua tabela, não necessariamente
# o limiar definitivo para uso operacional.
LIMIAR_ALERTA = 0.364

# Quantidade máxima de resultados exibidos individualmente no terminal.
# Os resultados de todos os registros serão salvos no CSV.
MAX_EXIBIR = 20


# ==============================================================================
# 2. CARREGAMENTO DOS MODELOS
# ==============================================================================

print("\n" + "=" * 72)
print(" SISTEMA DE PREVISÃO DE CARGA ALTA - CATBOOST")
print("=" * 72)

modelo1 = CatBoostClassifier()
modelo2 = CatBoostClassifier()

modelo1.load_model(caminho_modelo1)
modelo2.load_model(caminho_modelo2)

print("\nModelos carregados:")
print(f"  Cestão 1: {os.path.basename(caminho_modelo1)}")
print(f"  Cestão 2: {os.path.basename(caminho_modelo2)}")
print(f"  Limiar de alerta configurado: {LIMIAR_ALERTA:.1%}")


# ==============================================================================
# 3. LEITURA DA BASE
# ==============================================================================

try:
    dados = pd.read_csv(
        caminho_dados,
        sep=None,
        engine="python",
        encoding="utf-8"
    )
except UnicodeDecodeError:
    dados = pd.read_csv(
        caminho_dados,
        sep=None,
        engine="python",
        encoding="latin1"
    )

print(f"\nBase carregada: {dados.shape[0]} registros.")
print(f"Quantidade de colunas: {dados.shape[1]}")


# ==============================================================================
# 4. FUNÇÕES AUXILIARES
# ==============================================================================

def obter_indice_classe_alta(modelo):
    """
    Descobre em qual posição está a classe que representa carga alta.
    """

    classes = list(modelo.classes_)

    for i, classe in enumerate(classes):
        if str(classe) == str(CLASSE_CARGA_ALTA):
            return i

    raise ValueError(
        f"A classe {CLASSE_CARGA_ALTA} não foi encontrada no modelo. "
        f"Classes disponíveis: {classes}"
    )


def preparar_features(modelo, registros):
    """
    Seleciona as variáveis e preserva a ordem esperada pelo modelo.
    """

    colunas = list(modelo.feature_names_)

    if not colunas:
        raise ValueError(
            "O modelo não disponibilizou os nomes das features."
        )

    faltantes = [col for col in colunas if col not in registros.columns]

    if faltantes:
        raise ValueError(
            f"Faltam colunas exigidas pelo modelo: {faltantes}"
        )

    return registros[colunas]


def prever_cestao(modelo, registros, numero_cestao):
    """
    Calcula probabilidades, decisões e informações de cada registro.
    """

    if registros.empty:
        print(f"\nAviso: não existem registros para o cestão {numero_cestao}.")
        return pd.DataFrame()

    features = preparar_features(modelo, registros)

    indice_alta = obter_indice_classe_alta(modelo)
    probabilidades = modelo.predict_proba(features)

    prob_alta = probabilidades[:, indice_alta]
    prob_nao_alta = 1 - prob_alta

    alerta = prob_alta >= LIMIAR_ALERTA

    resultado = pd.DataFrame(index=registros.index)

    # Identificação do registro
    for coluna in ["cd_corrida", "nu_carregamento", "t"]:
        if coluna in registros.columns:
            resultado[coluna] = registros[coluna]

    resultado["modelo"] = f"CatBoost cestão {numero_cestao}"
    resultado["prob_carga_alta"] = prob_alta
    resultado["prob_carga_nao_alta"] = prob_nao_alta
    resultado["limiar_alerta"] = LIMIAR_ALERTA

    resultado["previsao"] = [
        "CARGA ALTA" if valor else "CARGA NÃO ALTA"
        for valor in alerta
    ]

    resultado["alerta"] = alerta

    # Inclui a classe real se estiver disponível na base.
    if "carga_alta" in registros.columns:
        resultado["carga_alta_real"] = registros["carga_alta"]

        resultado["avaliacao"] = [
            (
                "ACERTO"
                if str(real) == str(CLASSE_CARGA_ALTA) and previsto
                or str(real) != str(CLASSE_CARGA_ALTA) and not previsto
                else "ERRO"
            )
            for real, previsto in zip(
                registros["carga_alta"], alerta
            )
        ]

    # Outras informações operacionais disponíveis.
    for coluna in ["parada_s"]:
        if coluna in registros.columns:
            resultado[coluna] = registros[coluna]

    return resultado.reset_index(drop=True)


# ==============================================================================
# 5. SEPARAÇÃO E PREVISÃO DOS DOIS PRIMEIROS CESTÕES
# ==============================================================================

resultados = []

if "nu_carregamento" in dados.columns:
    # Evita misturar os registros do primeiro e do segundo cestão.
    dados_cestao1 = dados[
        pd.to_numeric(dados["nu_carregamento"], errors="coerce") == 1
    ].copy()

    dados_cestao2 = dados[
        pd.to_numeric(dados["nu_carregamento"], errors="coerce") == 2
    ].copy()

else:
    raise ValueError(
        "A base precisa conter a coluna nu_carregamento "
        "para separar os dois modelos corretamente."
    )

for modelo, registros, numero in [
    (modelo1, dados_cestao1, 1),
    (modelo2, dados_cestao2, 2)
]:
    resultado = prever_cestao(modelo, registros, numero)

    if not resultado.empty:
        resultados.append(resultado)


# ==============================================================================
# 6. CONSOLIDAÇÃO E EXPORTAÇÃO
# ==============================================================================

if not resultados:
    raise ValueError("Nenhuma previsão foi gerada.")

resultado_final = pd.concat(resultados, ignore_index=True)

resultado_final.to_csv(
    caminho_saida,
    index=False,
    encoding="utf-8-sig",
    sep=";"
)

print("\n" + "=" * 72)
print(" RESUMO DAS PREVISÕES")
print("=" * 72)

for numero in [1, 2]:
    grupo = resultado_final[
        resultado_final["modelo"] == f"CatBoost cestão {numero}"
    ]

    if grupo.empty:
        continue

    quantidade = len(grupo)
    alertas = int(grupo["alerta"].sum())

    print(f"\nCESTÃO {numero}")
    print(f"  Registros analisados: {quantidade}")
    print(f"  Alertas de carga alta: {alertas}")
    print(f"  Registros sem alerta: {quantidade - alertas}")
    print(f"  Fração de alertas: {alertas / quantidade:.1%}")

    if "carga_alta_real" in grupo.columns:
        print("  Comparação com os rótulos disponíveis:")
        print(
            grupo["avaliacao"]
            .value_counts()
            .to_string()
        )

print("\n" + "=" * 72)
print(f" PRIMEIROS {MAX_EXIBIR} RESULTADOS")
print("=" * 72)

colunas_exibir = [
    coluna for coluna in [
        "cd_corrida",
        "nu_carregamento",
        "prob_carga_alta",
        "prob_carga_nao_alta",
        "previsao",
        "carga_alta_real",
        "avaliacao"
    ]
    if coluna in resultado_final.columns
]

visualizacao = resultado_final[colunas_exibir].head(MAX_EXIBIR).copy()

for coluna in ["prob_carga_alta", "prob_carga_nao_alta"]:
    if coluna in visualizacao.columns:
        visualizacao[coluna] = visualizacao[coluna].map(
            lambda valor: f"{valor:.2%}"
        )

print(visualizacao.to_string(index=False))

print("\n" + "=" * 72)
print(" PREVISÕES CONCLUÍDAS")
print("=" * 72)
print(f"Arquivo completo salvo em:\n{caminho_saida}")