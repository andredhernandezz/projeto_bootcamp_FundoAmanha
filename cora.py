import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --- Carregando os tres datasets ---
materiais = pd.read_csv('infos_materiais.csv')
camadas = pd.read_csv('infos_camadas.csv')
resposta = pd.read_csv('VariavelResposta.csv')
print(f"materiais: {materiais.shape[0]} linhas x {materiais.shape[1]} colunas")
print(f"camadas: {camadas.shape[0]} linhas x {camadas.shape[1]} colunas")
print(f"resposta: {resposta.shape[0]} linhas x {resposta.shape[1]} colunas")