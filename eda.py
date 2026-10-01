import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# carregando os tres datasets
materiais = pd.read_csv('infos_materiais.csv', comment='#')
camadas = pd.read_csv('infos_camadas.csv')
resposta = pd.read_csv('VariavelResposta.csv')


# limpeza basica
# 2. Separar materiais metalicos de insumos
# Algumas linhas sao insumos (CAL, COQUE), nao sucata metalica
# Identifique-as: quais linhas tem null em Energia e Rendimento?
metais = materiais[materiais['Energia Elétrica'].notnull() & materiais['Rendimento Metálico %'].notnull()]
insumos = materiais[materiais['Energia Elétrica'].isnull() & materiais['Rendimento Metálico %'].isnull()]


# 1. Corrigir separador decimal em info_materiais
metais['Densidade t/m3'] = metais['Densidade t/m3'].str.replace(',', '.').astype(float)
metais['Rendimento Metálico %'] = metais['Rendimento Metálico %'].str.replace(',', '.').astype(float)


# tamanho dataset - linhas e colunas
print(f"materiais:  {materiais.shape[0]} linhas x {materiais.shape[1]} colunas")
print(f"camadas:    {camadas.shape[0]} linhas x {camadas.shape[1]} colunas")
print(f"resposta:   {resposta.shape[0]} linhas x {resposta.shape[1]} colunas")
print()

# inspecao rapida de cada dataset 
'''
for nome, df in [('materiais', materiais), ('camadas', camadas), ('resposta', resposta)]:
    print(f"\n{'='*50}")    # separador entre datasets
    print(f"    {nome.upper()}")
    print(f"{'='*50}")      # separador entre datasets
    print(df.info())
    print(f"\nPrimeiras linhas:\n{df.head()}")
    print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")
    print(f"\nEstatisticas descritivas:\n{df.describe()}")
'''

# 3.Converter datas
camadas['dt_hora_consumo'] = pd.to_datetime(camadas['dt_hora_consumo'])

# 4.Verificar duplicatas
print(f"num duplicatas em camadas: {(camadas.duplicated().sum())}")
print(f"num duplicatas em resposta: {resposta.duplicated(subset='cd_corrida').sum()}")

# 5.Quantas corridas unicas existem em cada dataset?
corridas_camadas = set(camadas['cd_corrida'].unique())
corridas_resposta = set(resposta['cd_corrida'].unique())

# Diferença entre conjuntos (A - B e B - A)
print(f"So em camadas: {len(corridas_camadas - corridas_resposta)}")
print(f"So em resposta: {len(corridas_resposta - corridas_camadas)}")

