import pandas as pd
import os
print(os.getcwd())

df = pd.read_csv("VariavelResposta.csv") # carrega um dataframe a partir do csv VariavelResposta
camadas = pd.read_csv("infos_camadas.csv", parse_dates=["dt_hora_consumo"])

# para cada corrida, pega a data mais antiga registrada
datas_corrida = (
    camadas.groupby("cd_corrida")["dt_hora_consumo"]
    .min()
    .reset_index()
    .rename(columns={"dt_hora_consumo": "dt_corrida"})
)

# junta a data com o dataframe principal
df = df.merge(datas_corrida, on="cd_corrida", how="left")

# checa se alguma corrida ficou sem data (não encontrada em infos_camadas)
print("Corridas sem data:", df["dt_corrida"].isna().sum())

# ordena cronologicamente
df = df.sort_values("dt_corrida").reset_index(drop=True)

# corte 90/10
n = len(df)
corte = int(n * 0.90)

treino = df.iloc[:corte]
teste = df.iloc[corte:]

print("Treino:", treino.shape, treino["dt_corrida"].min(), "→", treino["dt_corrida"].max())
print("Teste:", teste.shape, teste["dt_corrida"].min(), "→", teste["dt_corrida"].max())

treino.to_csv("treino.csv", index=False)
teste.to_csv("teste.csv", index=False)

df2 = pd.read_csv('infos_camadas.csv')
df2['dt_hora_consumo'] = pd.to_datetime(df2['dt_hora_consumo'])
df2.to_csv('infos_camadas_convertido.csv', index=False)