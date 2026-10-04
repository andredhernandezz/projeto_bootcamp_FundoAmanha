import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from scipy.stats import binomtest, chi2_contingency, spearmanr, mannwhitneyu, pointbiserialr
import itertools
from matplotlib.patches import Patch


# carregando os tres datasets
materiais = pd.read_csv('infos_materiais.csv', comment='#')
camadas = pd.read_csv('infos_camadas.csv')
resposta = pd.read_csv('VariavelResposta.csv')

####    PARTE 2 EDA ####

# limpeza basica
# 2. Separar materiais metalicos de insumos
# Algumas linhas sao insumos (CAL, COQUE), nao sucata metalica
# Identifique-as: quais linhas tem null em Energia e Rendimento?
metais = materiais[materiais['Energia Elétrica'].notnull() & materiais['Rendimento Metálico %'].notnull()]
insumos = materiais[materiais['Energia Elétrica'].isnull() & materiais['Rendimento Metálico %'].isnull()]


# 1. Corrigir separador decimal em info_materiais
metais['Densidade t/m3'] = pd.to_numeric(
    metais['Densidade t/m3'].astype(str).str.replace(',', '.'))
metais['Rendimento Metálico %'] = pd.to_numeric(
    metais['Rendimento Metálico %'].astype(str).str.replace(',', '.'))


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

# ============================================================
# SALVAR DE VOLTA NOS CSVs
# ============================================================

# garante que as colunas de destino aceitam valores numéricos (float) antes do .loc,
# já que nasceram como string por causa das linhas de insumo (CAL/COQUE)
materiais['Densidade t/m3'] = materiais['Densidade t/m3'].astype(object)
materiais['Rendimento Metálico %'] = materiais['Rendimento Metálico %'].astype(object)

# materiais: aplica as correções feitas em 'metais' no dataframe original
# (assim nenhuma linha se perde, nem as que não caíram em metais/insumos)
materiais.loc[metais.index, ['Densidade t/m3', 'Rendimento Metálico %']] = \
    metais[['Densidade t/m3', 'Rendimento Metálico %']]

materiais.to_csv('infos_materiais.csv', index=False)
camadas.to_csv('infos_camadas.csv', index=False, date_format='%Y-%m-%d %H:%M:%S')
resposta.to_csv('VariavelResposta.csv', index=False)



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
# ============================================================
# 3.0 CARREGAMENTO (robusto: pode rodar quantas vezes quiser)
# ============================================================
# dtype=str em 'Metálico' porque há códigos numéricos (7071955, 118441...)
raw = pd.read_csv('infos_materiais.csv', dtype={'Metálico': str})
 
col_en, col_dens, col_rend = 'Energia Elétrica', 'Densidade t/m3', 'Rendimento Metálico %'
 
# insumos = linhas sem energia e sem rendimento (CAL, COQUE)
eh_insumo = raw[col_en].isnull() & raw[col_rend].isnull()
 
# nas linhas de insumo, o nome (CAL/COQUE) está na coluna de densidade
insumos = (raw.loc[eh_insumo, ['Metálico', col_dens]]
              .rename(columns={'Metálico': 'material', col_dens: 'insumo'}))
 
metais = raw.loc[~eh_insumo].copy()
metais[col_dens] = pd.to_numeric(metais[col_dens])
metais = metais.rename(columns={
    'Metálico': 'material',
    col_en: 'energia',        # kWh/t
    col_dens: 'densidade',    # t/m3
    col_rend: 'rendimento',   # fração (0-1)
}).reset_index(drop=True)

# ============================================================
# 3.1 VISÃO GERAL
# ============================================================
print(f"Materiais metálicos: {len(metais)} | Insumos: {len(insumos)}")
print("\nInsumos:")
print(insumos.to_string(index=False))
 
print("\nTabela de materiais metálicos (ordenada por densidade):")
print(metais.sort_values('densidade').to_string(index=False))
 
 
def extremo(col, func):
    valor = getattr(metais[col], func)()
    nomes = metais.loc[metais[col] == valor, 'material'].tolist()
    return valor, nomes
 
 
for col, rotulo in [('densidade', 'densidade'), ('energia', 'energia elétrica'),
                    ('rendimento', 'rendimento')]:
    vmin, nmin = extremo(col, 'min')
    vmax, nmax = extremo(col, 'max')
    print(f"\n{rotulo}: MENOR = {vmin} {nmin} | MAIOR = {vmax} {nmax}")
 
# Spearman (ranking) porque a densidade tem outliers (GUSL/103089 = 7.4)
print("\nCorrelação de Spearman entre propriedades:")
print(metais[['densidade', 'energia', 'rendimento']].corr(method='spearman').round(2))
# p-valores das correlações de Spearman
print("\nSpearman com p-valor:")
for a, b in [('densidade', 'energia'),
             ('densidade', 'rendimento'),
             ('energia', 'rendimento')]:
    rho, p = spearmanr(metais[a], metais[b])
    print(f"{a:10s} x {b:10s} rho={rho:+.3f}  p={p:.4f}")

 # ============================================================
# 3.2 SCATTER: densidade x energia, cor = rendimento
# ============================================================
# materiais com propriedades idênticas ficam no mesmo ponto -> junta os rótulos
pontos = (metais.groupby(['densidade', 'energia', 'rendimento'])['material']
                .apply(lambda s: '/'.join(s)).reset_index())
 
fig, ax = plt.subplots(figsize=(11, 6.5))
sc = ax.scatter(pontos['densidade'], pontos['energia'], c=pontos['rendimento'],
                cmap='RdYlGn', s=110, edgecolors='k')
for _, r in pontos.iterrows():
    ax.annotate(r['material'], (r['densidade'], r['energia']),
                fontsize=8, xytext=(4, 4), textcoords='offset points')
ax.set_xscale('log')  # densidade vai de 0.3 a 7.4
ax.set_xticks([0.3, 0.5, 1, 2, 3, 7])
ax.xaxis.set_major_formatter(ScalarFormatter())
plt.colorbar(sc, label='Rendimento metálico')
ax.set_xlabel('Densidade (t/m³) - escala log')
ax.set_ylabel('Energia elétrica (kWh/t)')
ax.set_title('Propriedades dos materiais metálicos')
plt.tight_layout()
plt.savefig('fig_3_2_scatter_materiais.png', dpi=150)
# plt.show() # PLT SHOW ESTA COMENTADO APENAS PARA NAO APARECER -----------------------------------------------------------------------------------
 
# ============================================================
# 3.3 CLASSIFICAÇÃO POR CLASSE DE PESO (listas do guia)
# ============================================================
leves = ['SHRE', 'ESPS', 'CAVA', 'RETL', 'PACB', 'ESBS', 'MARP', 'MISS']
pesados = ['GUSS', 'GUSL', 'RECA', 'SUGU', 'FOFO', '194032', '118441']
mistos = ['OXIC', 'MISM', 'PAEP', 'MISB', 'CAFO', 'ESPR', 'PAEB']
 
mapa = {**{m: 'leve' for m in leves},
        **{m: 'pesado' for m in pesados},
        **{m: 'misto' for m in mistos}}
 
# Materiais que NÃO estão nas listas do guia: decidir com o orientador.
# Exemplo de como atribuir depois de decidir:
# mapa.update({'PACH': 'leve', 'GUSD': 'pesado', '103089': 'pesado'})
metais['classe'] = metais['material'].map(mapa).fillna('sem_classe')
 
print("\nMateriais fora das listas do guia:")
print(metais.loc[metais['classe'] == 'sem_classe',
                 ['material', 'densidade', 'energia', 'rendimento']].to_string(index=False))
 
resumo = (metais.groupby('classe')
                .agg(n=('material', 'count'),
                     dens_media=('densidade', 'mean'),
                     dens_min=('densidade', 'min'),
                     dens_max=('densidade', 'max'),
                     energia_media=('energia', 'mean'),
                     rend_medio=('rendimento', 'mean'))
                .round(2))
print("\nResumo por classe:")
print(resumo)
 
# A classificação do guia faz sentido físico? Quem fica fora do esperado:
print("\nLEVES com densidade >= 0.55 ou rendimento <= 0.90:")
print(metais[(metais['classe'] == 'leve') &
             ((metais['densidade'] >= 0.55) | (metais['rendimento'] <= 0.90))]
      [['material', 'densidade', 'rendimento']].to_string(index=False))
print("\nPESADOS com densidade <= 2.5:")
print(metais[(metais['classe'] == 'pesado') & (metais['densidade'] <= 2.5)]
      [['material', 'densidade']].to_string(index=False))
 
# Densidade por material, colorida por classe
ordem = metais.sort_values('densidade')
fig, ax = plt.subplots(figsize=(8, 9))
sns.barplot(data=ordem, y='material', x='densidade', hue='classe', dodge=False, ax=ax,
            palette={'leve': 'steelblue', 'misto': 'goldenrod',
                     'pesado': 'firebrick', 'sem_classe': 'gray'})
ax.set_xlabel('Densidade (t/m³)')
ax.set_ylabel('')
ax.set_title('Densidade por material e classe')
plt.tight_layout()
plt.savefig('fig_3_3_densidade_por_classe.png', dpi=150)
# plt.show() # PLT SHOW ESTA COMENTADO APENAS PARA NAO APARECER -----------------------------------------------------------------------------------
 
# Arquivo novo (não sobrescreve os originais) para as próximas etapas
metais.to_csv('materiais_classificados.csv', index=False)
print("\nSalvo: materiais_classificados.csv")

 
# Investigue:
# - Quantos materiais metalicos existem? E quantos insumos?
    # são 27 materiais metálicos e 3 insumos (CAL e dois COQUE).
# - Qual material tem a MENOR densidade? E a MAIOR?
    # a menor é MISS (0,30 t/m³) e a maior é GUSL (7,4, empatada com 103089).
# - Qual material consome mais energia por tonelada?
    # o maior consumo é de CAVA e MARP (405 kWh/t); o menor é 293 (GUSD, 118441, GUSL e 103089).
# - Ha alguma relacao entre densidade e rendimento metalico?
    # a relação é fraca (Spearman 0,25). A relação mais forte é energia × rendimento, negativa (−0,54): quanto mais energia, menor o rendimento.

""""
Cinco materiais não estão em nenhuma lista do guia: PACH, GUSD, FORJ, EXTR e 103089. Ficaram como sem_classe. Pelas propriedades, PACH parece leve (igual ao PACB), GUSD e 103089 parecem pesados, e FORJ parece misto (1,2 t/m³, como ESPR e MISB). EXTR (0,70) é ambíguo. Como eles provavelmente aparecem nas camadas, a classificação precisa ser fechada antes da etapa 4. A linha mapa.update(...) comentada no script mostra onde atribuir.
O "insight esperado" do guia não vale para os leves. Ele diz que os leves têm rendimento acima de 0,90, mas só 3 dos 8 passam disso. PACB, CAVA, MARP e MISS ficam entre 0,83 e 0,87. SHRE tem densidade 0,70, acima do limite de 0,55 do guia. Para os pesados, a regra de densidade acima de 2,5 quase vale (SUGU tem exatamente 2,5). A média de rendimento é 0,89 nas três classes, então rendimento não separa as classes. A densidade separa bem: 0,48 nos leves, 0,94 nos mistos e 3,48 nos pesados.
GUSL e 103089 têm densidade 7,4, bem acima do resto. Isso é plausível para ferro-gusa líquido, mas distorce o scatter, por isso usei escala logarítmica no eixo x. Como o carregamento 3 é só gusa líquido e não causa carga alta, vale conferir depois se ele entra no cálculo de volume.
"""



# --- Estrutura dos carregamentos ---
# Entenda a hierarquia: Corrida -> Carregamento -> Camada
print("Carregamentos por corrida:")
print(camadas.groupby('cd_corrida')['nu_carregamento'].nunique().describe())
print("\nCamadas por carregamento:")
print(camadas.groupby(['cd_corrida', 'nu_carregamento'])['nu_camada'].count().describe())
# Investigue:
# - Todas as corridas tem 3 carregamentos (2 cestos + gusa)?
# - Quantas camadas tem, em media, cada carregamento?
# - O carregamento 3 e sempre GUSL? Verifique.



# ============================================================
# 4.0 CAMADAS E CARREGAMENTO 
# ============================================================
CHAVE = ['cd_corrida', 'nu_carregamento']          # um "cestão" = corrida + carregamento
CORES_CLASSE = {'leve': 'steelblue', 'misto': 'goldenrod', 'pesado': 'firebrick',
                'sem_classe': 'gray', 'fora_dicionario': 'lightgray'}   # mesmas cores da etapa 3
CORES_CARREG = {1: '#000000', 2: '#f63f01'}


def titulo(txt):
    print(f"\n{'=' * 60}\n{txt}\n{'=' * 60}")


def estilo(ax, grade='y'):
    """Eixos limpos: sem moldura em cima/direita e grade discreta."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=grade, alpha=0.25)
    ax.set_axisbelow(True)


# ============================================================
# 4.0 CARREGAMENTO E PREPARO (em memória; nada é regravado)
# ============================================================
camadas = pd.read_csv('infos_camadas.csv', dtype={'cd_codigo_material': str})
materiais = pd.read_csv('materiais_classificados.csv', dtype={'material': str})
resposta = pd.read_csv('VariavelResposta.csv')

n0 = len(camadas)
camadas = camadas.drop_duplicates()
print(f"Duplicatas exatas removidas de camadas: {n0 - len(camadas)}")

camadas['dt_hora_consumo'] = pd.to_datetime(camadas['dt_hora_consumo'])
camadas['nu_carregamento'] = camadas['nu_carregamento'].astype(int)
camadas['nu_camada'] = camadas['nu_camada'].astype(int)
camadas['peso_t'] = camadas['qt_peso_carregamento'] / 1000      # kg -> toneladas

# classe de cada material (listas do guia, etapa 3)
mapa_classe = materiais.set_index('material')['classe'].to_dict()
camadas['classe'] = camadas['cd_codigo_material'].map(mapa_classe).fillna('fora_dicionario')

# cobertura: dicionário x camadas
mat_camadas = set(camadas['cd_codigo_material'])
mat_dicionario = set(materiais['material'])
print("Materiais nas camadas FORA do dicionário:", sorted(mat_camadas - mat_dicionario))
print("Materiais do dicionário que NUNCA aparecem nas camadas:", sorted(mat_dicionario - mat_camadas))
sem_classe = set(materiais.loc[materiais['classe'] == 'sem_classe', 'material'])
print("Dos 'sem_classe' (etapa 3), aparecem nas camadas:", sorted(sem_classe & mat_camadas) or "nenhum")
fora = camadas[camadas['classe'] == 'fora_dicionario']
print(fora.groupby(['cd_codigo_material', 'ds_descricao_material'])
          .agg(linhas=('peso_t', 'size'), peso_t=('peso_t', 'sum')).round(1))

# ============================================================
# 4.1 ESTRUTURA DOS CARREGAMENTOS
# ============================================================
titulo("4.1 ESTRUTURA: corrida -> carregamento -> camada")

combo = (camadas.groupby('cd_corrida')['nu_carregamento']
                .apply(lambda s: '+'.join(str(x) for x in sorted(s.unique()))))
print("Combinações de carregamentos por corrida (nº de corridas):")
print(combo.value_counts().to_string())

n_camadas = camadas.groupby(CHAVE)['nu_camada'].nunique()
print("\nCamadas por carregamento:")
print(n_camadas.groupby(level='nu_carregamento').describe().round(1))

c3 = camadas[camadas['nu_carregamento'] == 3]
print("\nMateriais no carregamento 3 (nº de linhas):")
print(c3['cd_codigo_material'].value_counts().to_string())
print(f"Corridas com carregamento 3: {c3['cd_corrida'].nunique()} | "
      f"com GUSL: {c3.loc[c3['cd_codigo_material'] == 'GUSL', 'cd_corrida'].nunique()}")
print("Peso do carregamento 3 por corrida (t):")
print((c3.groupby('cd_corrida')['peso_t'].sum()).describe().round(1))

# a partir daqui: só os cestões sólidos (carregamentos 1 e 2)
solidos = camadas[camadas['nu_carregamento'].isin([1, 2])].copy()
cestoes = (solidos.groupby(CHAVE)
           .agg(peso_total_t=('peso_t', 'sum'),
                num_camadas=('nu_camada', 'nunique'),
                num_materiais=('cd_codigo_material', 'nunique'),
                camadas_peso_zero=('peso_t', lambda s: int((s == 0).sum())))
           .reset_index())
print(f"\nCestões sólidos (corrida x carregamento 1/2): {len(cestoes)}")

fig, ax = plt.subplots(figsize=(9, 5))
sns.countplot(data=cestoes, x='num_camadas', hue='nu_carregamento',
              palette=CORES_CARREG, ax=ax)
ax.set_xlabel('Número de camadas no cestão')
ax.set_ylabel('Nº de cestões')
ax.set_title('Camadas por cestão, por carregamento')
ax.legend(title='Carregamento')
estilo(ax)
plt.tight_layout()
plt.savefig('fig_4_1_camadas_por_cestao.png', dpi=150)
plt.show()

# ============================================================
# 4.2 DISTRIBUIÇÃO DE PESO POR CARREGAMENTO
# ============================================================
titulo("4.2 PESO TOTAL POR CESTÃO (toneladas)")

print(cestoes.groupby('nu_carregamento')['peso_total_t'].describe().round(1))
med = cestoes.groupby('nu_carregamento')['peso_total_t'].median()
print(f"\nMediana: carregamento 1 = {med[1]:.1f} t | carregamento 2 = {med[2]:.1f} t "
      f"-> o carregamento {1 if med[1] > med[2] else 2} é mais pesado")

for c, g in cestoes.groupby('nu_carregamento'):
    q1, q3 = g['peso_total_t'].quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    abaixo, acima = g[g['peso_total_t'] < lo], g[g['peso_total_t'] > hi]
    print(f"\nCarregamento {c}: faixa típica (Q1-Q3) {q1:.1f}-{q3:.1f} t | "
          f"outliers: {len(abaixo)} abaixo de {lo:.1f} t e {len(acima)} acima de {hi:.1f} t")
    extremos = pd.concat([abaixo.nsmallest(3, 'peso_total_t'), acima.nlargest(3, 'peso_total_t')])
    if len(extremos):
        print(extremos[['cd_corrida', 'peso_total_t', 'num_camadas']].round(1).to_string(index=False))

# o peso do cestão muda quando a corrida também recebe gusa líquido (carregamento 3)?
tem_c3 = (camadas.groupby('cd_corrida')['nu_carregamento']
                 .apply(lambda s: bool((s == 3).any())).rename('tem_gusa_liquido'))
cestoes = cestoes.merge(tem_c3, left_on='cd_corrida', right_index=True, how='left')
print("\nPeso total (t) por carregamento, com e sem carregamento 3 (gusa líquido) na corrida:")
print(cestoes.groupby(['nu_carregamento', 'tem_gusa_liquido'])['peso_total_t']
      .agg(cestoes='count', media='mean', mediana='median', maximo='max').round(1).to_string())

zeros = solidos[solidos['peso_t'] == 0]
print(f"\nCamadas com peso 0: {len(zeros)} ({len(zeros) / len(solidos):.1%} das camadas dos cestões) "
      f"em {(cestoes['camadas_peso_zero'] > 0).sum()} cestões")
print("Materiais mais comuns entre as camadas de peso 0:")
print(zeros['cd_codigo_material'].value_counts().head(5).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=cestoes, x='nu_carregamento', y='peso_total_t', hue='nu_carregamento',
            palette=CORES_CARREG, legend=False, width=0.5, fliersize=3, ax=axes[0])
axes[0].set_xlabel('Carregamento')
axes[0].set_ylabel('Peso total do cestão (t)')
axes[0].set_title('Peso total por carregamento')
estilo(axes[0])
for c in (1, 2):
    axes[1].hist(cestoes.loc[cestoes['nu_carregamento'] == c, 'peso_total_t'],
                 bins=30, alpha=0.65, color=CORES_CARREG[c], label=f'Carregamento {c}')
axes[1].set_xlabel('Peso total do cestão (t)')
axes[1].set_ylabel('Nº de cestões')
axes[1].set_title('Distribuição do peso total')
axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.2)      # folga no topo para a legenda não cobrir as barras
axes[1].legend(loc='upper center', ncol=2)
estilo(axes[1])
plt.tight_layout()
plt.savefig('fig_4_2_peso_por_carregamento.png', dpi=150)
plt.show()

# ============================================================
# 4.3 COMPOSIÇÃO POR MATERIAL
# ============================================================
titulo("4.3 COMPOSIÇÃO POR MATERIAL (carregamentos 1 e 2)")

n_cestoes_total = len(cestoes)
cestoes_por_mat = (solidos.drop_duplicates(CHAVE + ['cd_codigo_material'])
                          .groupby('cd_codigo_material').size())
por_material = (solidos.groupby('cd_codigo_material')
                .agg(peso_total_t=('peso_t', 'sum'),
                     n_camadas=('nu_camada', 'size'),
                     peso_medio_camada_t=('peso_t', 'mean'))
                .assign(n_cestoes=cestoes_por_mat,
                        pct_cestoes=lambda d: d['n_cestoes'] / n_cestoes_total * 100,
                        pct_peso=lambda d: d['peso_total_t'] / d['peso_total_t'].sum() * 100))
por_material['classe'] = [mapa_classe.get(m, 'fora_dicionario') for m in por_material.index]
por_material = por_material.sort_values('peso_total_t', ascending=False)
print(por_material.round(1).to_string())
print(f"\nMaior peso total: {por_material.index[0]} ({por_material['pct_peso'].iloc[0]:.1f}% do peso) | "
      f"mais camadas: {por_material['n_camadas'].idxmax()} ({por_material['n_camadas'].max()}) | "
      f"em mais cestões: {por_material['n_cestoes'].idxmax()} "
      f"({por_material['pct_cestoes'].max():.1f}% dos cestões)")

ordem = por_material.index.tolist()
cores = [CORES_CLASSE[c] for c in por_material['classe']]
fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
axes[0].barh(ordem, por_material['peso_total_t'], color=cores)
axes[0].set_xlabel('Peso total consumido (t)')
axes[0].set_title('Peso total por material')
axes[1].barh(ordem, por_material['pct_cestoes'], color=cores)
axes[1].set_xlabel('% dos cestões em que o material aparece')
axes[1].set_title('Frequência por material')
axes[0].invert_yaxis()
for ax in axes:
    estilo(ax, grade='x')
classes_presentes = [c for c in CORES_CLASSE if c in set(por_material['classe'])]
axes[1].legend(handles=[Patch(color=CORES_CLASSE[c], label=c) for c in classes_presentes],
               title='Classe', loc='lower right')
plt.tight_layout()
plt.savefig('fig_4_3_composicao_material.png', dpi=150)
plt.show()

# carregamento 1 x carregamento 2 (participação percentual no peso)
comp = solidos.pivot_table(index='cd_codigo_material', columns='nu_carregamento',
                           values='peso_t', aggfunc='sum', fill_value=0)
comp_pct = comp.div(comp.sum()) * 100
comp_pct.columns = ['carreg_1_pct', 'carreg_2_pct']
comp_pct['dif_2_menos_1'] = comp_pct['carreg_2_pct'] - comp_pct['carreg_1_pct']
print("\nMaiores diferenças de composição (% do peso) entre carregamento 2 e 1:")
print(comp_pct.reindex(comp_pct['dif_2_menos_1'].abs().sort_values(ascending=False).index)
              .head(8).round(1).to_string())

top = (comp_pct[comp_pct[['carreg_1_pct', 'carreg_2_pct']].max(axis=1) >= 1]
       .sort_values('carreg_1_pct', ascending=False))
y, h = np.arange(len(top)), 0.38
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(y - h / 2, top['carreg_1_pct'], height=h, color=CORES_CARREG[1], label='Carregamento 1')
ax.barh(y + h / 2, top['carreg_2_pct'], height=h, color=CORES_CARREG[2], label='Carregamento 2')
ax.set_yticks(y)
ax.set_yticklabels(top.index)
ax.invert_yaxis()
ax.set_xlabel('% do peso do carregamento')
ax.set_title('Composição por material: carregamento 1 x 2 (materiais com 1% ou mais)')
ax.legend()
estilo(ax, grade='x')
plt.tight_layout()
plt.savefig('fig_4_3b_carreg1_vs_2.png', dpi=150)
plt.show()

# materiais que (quase) nunca aparecem juntos no mesmo cestão
presenca = (solidos.assign(v=1)
                   .pivot_table(index=CHAVE, columns='cd_codigo_material', values='v',
                                aggfunc='max', fill_value=0))
co = presenca.T.dot(presenca)
freq = presenca.sum()
frequentes = freq[freq >= 100].index.tolist()
linhas = []
for a, b in itertools.combinations(frequentes, 2):
    juntos = int(co.loc[a, b])
    linhas.append((a, b, juntos, juntos / (freq[a] + freq[b] - juntos)))
pares = (pd.DataFrame(linhas, columns=['mat_a', 'mat_b', 'cestoes_juntos', 'jaccard'])
           .sort_values('jaccard'))
print(f"\nPares (entre {len(frequentes)} materiais presentes em 100+ cestões) "
      f"que NUNCA aparecem juntos: {(pares['cestoes_juntos'] == 0).sum()}")
print("Os 8 pares que menos aparecem juntos (Jaccard = juntos / pelo menos um dos dois):")
print(pares.head(8).round(3).to_string(index=False))

# ============================================================
# 4.4 ORDEM E POSIÇÃO DAS CAMADAS (regra leve - pesado - leve)
# ============================================================
titulo("4.4 POSIÇÃO DAS CAMADAS")

solidos['n_camadas_cestao'] = solidos.groupby(CHAVE)['nu_camada'].transform('max')
solidos['pos_relativa'] = solidos['nu_camada'] / solidos['n_camadas_cestao']   # 1 = fundo ... 1.0 = topo
solidos['zona'] = pd.cut(solidos['pos_relativa'], bins=[0, 0.3, 0.7, 1.0],
                         labels=['fundo', 'meio', 'topo'], include_lowest=True)

pos_classe = solidos.groupby('classe')['pos_relativa'].agg(camadas='count', media='mean', mediana='median')
zona_classe = pd.crosstab(solidos['classe'], solidos['zona'], normalize='index') * 100
print("Posição relativa por classe (0 = fundo, 1 = topo):")
print(pos_classe.round(2).to_string())
print("\n% das camadas de cada classe em cada zona (fundo <= 0,3 < meio <= 0,7 < topo):")
print(zona_classe.round(1).to_string())

pos_mat = (solidos.groupby('cd_codigo_material')
           .agg(camadas=('pos_relativa', 'size'), pos_media=('pos_relativa', 'mean'),
                classe=('classe', 'first'))
           .sort_values('pos_media'))
print("\nPosição média por material (materiais com 100+ camadas):")
print(pos_mat[pos_mat['camadas'] >= 100].round(2).to_string())

tres = ['leve', 'misto', 'pesado']
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=solidos[solidos['classe'].isin(tres)], x='classe', y='pos_relativa',
            order=tres, hue='classe', palette=CORES_CLASSE, legend=False, width=0.5,
            fliersize=2, ax=axes[0])
for lim in (0.3, 0.7):
    axes[0].axhline(lim, color='gray', ls='--', lw=1)
axes[0].set_xlabel('Classe')
axes[0].set_ylabel('Posição relativa no cestão (0 = fundo, 1 = topo)')
axes[0].set_title('Onde cada classe é colocada')
estilo(axes[0])

zonas = zona_classe.loc[tres, ['fundo', 'meio', 'topo']]
cores_zona = {'fundo': '#9ecae1', 'meio': '#4292c6', 'topo': '#08519c'}
esq = np.zeros(len(zonas))
for z in zonas.columns:
    axes[1].barh(zonas.index, zonas[z], left=esq, color=cores_zona[z], label=z, height=0.55)
    for i, (v, l) in enumerate(zip(zonas[z], esq)):
        if v >= 8:
            axes[1].text(l + v / 2, i, f"{v:.0f}%", ha='center', va='center',
                         color='white' if z != 'fundo' else 'black', fontsize=9)
    esq = esq + zonas[z].values
axes[1].invert_yaxis()
axes[1].set_xlabel('% das camadas da classe')
axes[1].set_title('Distribuição por zona do cestão')
axes[1].legend(title='Zona', ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.15))
estilo(axes[1], grade='x')
plt.tight_layout()
plt.savefig('fig_4_4_posicao_camadas.png', dpi=150)
plt.show()

# --- regra leve-pesado-leve por cestão ---------------------------------
base_leve = (solidos[solidos['nu_camada'] == 1]
             .groupby(CHAVE)['classe'].apply(lambda s: bool((s == 'leve').any())).rename('base_leve'))
topo_leve = (solidos[solidos['nu_camada'] == solidos['n_camadas_cestao']]
             .groupby(CHAVE)['classe'].apply(lambda s: bool((s == 'leve').any())).rename('topo_leve'))
leve_no_meio = (solidos[solidos['classe'] == 'leve']
                .groupby(CHAVE)['zona'].apply(lambda s: float((s == 'meio').mean()))
                .rename('leve_no_meio_pct'))
pesado_fora_meio = (solidos[solidos['classe'] == 'pesado']
                    .groupby(CHAVE)['zona'].apply(lambda s: float((s != 'meio').mean()))
                    .rename('pesado_fora_meio_pct'))
cestoes = (cestoes.set_index(CHAVE)
           .join([base_leve, topo_leve, leve_no_meio, pesado_fora_meio])
           .reset_index())
cestoes[['base_leve', 'topo_leve']] = cestoes[['base_leve', 'topo_leve']].fillna(False).astype(bool)
cestoes['regra_ok'] = cestoes['base_leve'] & cestoes['topo_leve']

print("\nRegra 'base e topo leves' (material leve na camada 1 e na última camada):")
print(cestoes.groupby('nu_carregamento')[['base_leve', 'topo_leve', 'regra_ok']].mean().mul(100).round(1)
      .rename(columns=lambda c: c + ' (%)').to_string())

# cruza com a resposta (um indicador por cestão) - checagem preliminar, a análise completa é a seção 6
r_long = pd.concat([
    pd.DataFrame({'cd_corrida': resposta['cd_corrida'], 'nu_carregamento': 1,
                  'carga_alta': resposta['is_carga_alta_carregamento_1'],
                  'parada_s': resposta['qt_segundos_duracao_parada_carregamento_1']}),
    pd.DataFrame({'cd_corrida': resposta['cd_corrida'], 'nu_carregamento': 2,
                  'carga_alta': resposta['is_carga_alta_carregamento_2'],
                  'parada_s': resposta['qt_segundos_duracao_parada_carregamento_2']}),
])
cestoes = cestoes.merge(r_long, on=CHAVE, how='left')
print(f"\nCestões sem linha correspondente na variável resposta: {cestoes['carga_alta'].isnull().sum()}")

print("\nTaxa de carga alta por cumprimento da regra (preliminar):")
print(cestoes.groupby(['nu_carregamento', 'regra_ok'])['carga_alta']
      .agg(taxa='mean', n='size').round(3).to_string())
# testes SEPARADOS por carregamento: os cestões 1 e 2 têm pesos e taxas bem diferentes,
# então juntar os dois misturaria o efeito do carregamento com o efeito da regra
print("\nQui-quadrado (regra x carga alta), por carregamento:")
for c, g in cestoes.groupby('nu_carregamento'):
    for col in ['base_leve', 'topo_leve', 'regra_ok']:
        tab = pd.crosstab(g[col], g['carga_alta'])
        if tab.shape != (2, 2):
            print(f"  carregamento {c} | {col:10s} sem variação suficiente para testar")
            continue
        _, p, _, _ = chi2_contingency(tab)
        taxas = g.groupby(col)['carga_alta'].mean().round(3).to_dict()
        print(f"  carregamento {c} | {col:10s} taxa {taxas} | p = {p:.4f}")

print("\nMédias por resultado do cestão (carga_alta 0 x 1), por carregamento - prévia da seção 6:")
print(cestoes.groupby(['nu_carregamento', 'carga_alta'])
      [['peso_total_t', 'num_camadas', 'num_materiais',
        'leve_no_meio_pct', 'pesado_fora_meio_pct']].mean().round(2).to_string())

# ============================================================
# SALVAR (arquivos novos; os de entrada não são tocados)
# ============================================================
cestoes.to_csv('cestoes_resumo.csv', index=False)
por_material.to_csv('composicao_por_material.csv')
print("\nSalvos: cestoes_resumo.csv, composicao_por_material.csv")


# ============================================================
# 5.0 VARIÁVEL RESPOSTA (CARGA ALTA)
# ============================================================


# Custo de cada minuto de parada (R$/min). Definir com o orientador; enquanto for None,
# o script mostra só os minutos perdidos.
CUSTO_POR_MINUTO = None

COR_SEM, COR_COM = 'steelblue', 'tomato'
CORES_CARREG = {1: '#000000', 2: '#f63f01'}      # mesmas cores da etapa 4


def titulo(txt):
    print(f"\n{'=' * 60}\n{txt}\n{'=' * 60}")


def estilo(ax, grade='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis=grade, alpha=0.25)
    ax.set_axisbelow(True)


# ============================================================
# 5.0 CARREGAMENTO E CONSISTÊNCIA
# ============================================================
resp = pd.read_csv('VariavelResposta.csv')
camadas = pd.read_csv('infos_camadas.csv', usecols=['cd_corrida', 'nu_carregamento', 'dt_hora_consumo'])
camadas['dt_hora_consumo'] = pd.to_datetime(camadas['dt_hora_consumo'])

f1 = resp['is_carga_alta_carregamento_1']
f2 = resp['is_carga_alta_carregamento_2']
d1 = resp['qt_segundos_duracao_parada_carregamento_1']
d2 = resp['qt_segundos_duracao_parada_carregamento_2']
flag = resp['is_carga_alta']
dur = resp['qt_segundos_parada_carga_alta']

titulo("5.0 CONSISTÊNCIA DA VARIÁVEL RESPOSTA")
print(f"Corridas: {len(resp)} | duplicadas por cd_corrida: {resp.duplicated(subset='cd_corrida').sum()} "
      f"| nulos: {int(resp.isnull().sum().sum())}")
print("is_carga_alta == (carregamento 1 OU 2)?      ", bool((flag == (f1 | f2)).all()))
print("parada total == parada 1 + parada 2?         ", bool((dur == d1 + d2).all()))
print("eventos (flag=1) com duração 0 -> c1 / c2:   ", int(((f1 == 1) & (d1 == 0)).sum()),
      "/", int(((f2 == 1) & (d2 == 0)).sum()))
print("sem evento (flag=0) com duração > 0 -> c1 / c2:", int(((f1 == 0) & (d1 > 0)).sum()),
      "/", int(((f2 == 0) & (d2 > 0)).sum()))

# ============================================================
# 5.1 DISTRIBUIÇÃO DO EVENTO
# ============================================================
titulo("5.1 DISTRIBUIÇÃO DE is_carga_alta")

cont = flag.value_counts().sort_index()
taxa = flag.mean()
print(cont.to_string())
print(f"\nTaxa de carga alta: {taxa:.1%} ({int(cont[1])} de {len(resp)} corridas)")
print(f"Proporção sem:com evento = {cont[0] / cont[1]:.1f} : 1")
print("Leitura: o guia chama de muito desbalanceado abaixo de ~20%; aqui a classe positiva "
      "é minoria (≈30%), um desbalanceamento moderado. Na modelagem, usar split estratificado e "
      "métricas além da acurácia (precisão, recall, F1, PR-AUC).")

fig, ax = plt.subplots(figsize=(6.5, 5))
barras = ax.bar(['Sem carga alta', 'Com carga alta'], cont.values, color=[COR_SEM, COR_COM], width=0.55)
ax.bar_label(barras, labels=[f"{v}  ({v / len(resp):.1%})" for v in cont.values], padding=3)
ax.set_ylabel('Nº de corridas')
ax.set_title('Corridas com e sem carga alta')
ax.set_ylim(0, cont.max() * 1.12)
estilo(ax)
plt.tight_layout()
plt.savefig('fig_5_1_distribuicao_evento.png', dpi=150)
plt.show()

# ============================================================
# 5.2 CARGA ALTA POR CARREGAMENTO
# ============================================================
titulo("5.2 CARGA ALTA POR CARREGAMENTO")

tem_c2 = resp['cd_corrida'].isin(set(camadas.loc[camadas['nu_carregamento'] == 2, 'cd_corrida']))
print(f"Corridas SEM cestão 2 nas camadas: {(~tem_c2).sum()} | carga alta no carregamento 2 nelas: "
      f"{int(f2[~tem_c2].sum())}")

taxa1 = f1.mean()
taxa2 = f2[tem_c2].mean()
print(f"\nTaxa no carregamento 1: {taxa1:.1%} (base: {len(resp)} corridas)")
print(f"Taxa no carregamento 2: {taxa2:.1%} (base: {int(tem_c2.sum())} corridas que têm cestão 2)")
print(f"[Para comparar com o guia, usando todas as {len(resp)} corridas: "
      f"carregamento 1 = {f1.mean():.1%}, carregamento 2 = {f2.mean():.1%}]")

resp['combo'] = np.select([(f1 == 1) & (f2 == 1), (f1 == 1) & (f2 == 0), (f1 == 0) & (f2 == 1)],
                          ['ambos', 'só carreg. 1', 'só carreg. 2'], default='nenhum')
ordem_combo = ['nenhum', 'só carreg. 1', 'só carreg. 2', 'ambos']
combo = resp['combo'].value_counts().reindex(ordem_combo)
print("\nCorridas por combinação de carga alta:")
print(pd.DataFrame({'corridas': combo, 'pct': (combo / len(resp) * 100).round(1)}).to_string())

# comparação entre carregamentos 1 e 2 nas corridas que têm os dois cestões
par = resp[tem_c2]
p1, p2 = par['is_carga_alta_carregamento_1'], par['is_carga_alta_carregamento_2']
b = int(((p1 == 1) & (p2 == 0)).sum())      # só no 1
c = int(((p1 == 0) & (p2 == 1)).sum())      # só no 2
mc = binomtest(b, b + c, 0.5).pvalue
print(f"\nNas {len(par)} corridas com os dois cestões: carregamento 1 = {p1.mean():.1%}, "
      f"carregamento 2 = {p2.mean():.1%}")
print(f"Teste de McNemar exato (só no 1 = {b}, só no 2 = {c}): "
      f"p {'< 0.0001' if mc < 0.0001 else f'= {mc:.4f}'}")

tab = pd.crosstab(p1, p2)
chi, p_ind, _, _ = chi2_contingency(tab)
cond = par.groupby('is_carga_alta_carregamento_1')['is_carga_alta_carregamento_2'].mean()
print(f"\nSe houve carga alta no 1, a chance de carga alta no 2 é {cond[1]:.1%}; "
      f"se não houve, {cond[0]:.1%} (qui-quadrado de independência p = {p_ind:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
barras = axes[0].bar(['Carregamento 1', 'Carregamento 2'], [taxa1 * 100, taxa2 * 100],
                     color=[CORES_CARREG[1], CORES_CARREG[2]], width=0.5)
axes[0].bar_label(barras, labels=[f"{taxa1:.1%}\n(n = {len(resp)})",
                                  f"{taxa2:.1%}\n(n = {int(tem_c2.sum())})"], padding=3)
axes[0].set_ylabel('% de corridas com carga alta')
axes[0].set_ylim(0, max(taxa1, taxa2) * 100 * 1.3)
axes[0].set_title('Taxa de carga alta por carregamento')
estilo(axes[0])

cores_combo = [COR_SEM, COR_COM, COR_COM, COR_COM]
barras = axes[1].bar(ordem_combo, combo.values, color=cores_combo, width=0.6)
axes[1].bar_label(barras, labels=[f"{v}\n({v / len(resp):.1%})" for v in combo.values], padding=3)
axes[1].set_ylabel('Nº de corridas')
axes[1].set_ylim(0, combo.max() * 1.15)
axes[1].set_title('Em qual carregamento ocorreu a carga alta')
estilo(axes[1])
plt.tight_layout()
plt.savefig('fig_5_2_carga_alta_por_carregamento.png', dpi=150)
plt.show()

# ============================================================
# 5.3 DURAÇÃO DAS PARADAS
# ============================================================
titulo("5.3 DURAÇÃO DAS PARADAS (somente corridas com carga alta)")

ev = dur[flag == 1]
print("Duração da parada por evento (segundos):")
print(ev.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]).round(0).to_string())
print(f"\nMédia = {ev.mean():.0f} s ({ev.mean() / 60:.1f} min) | mediana = {ev.median():.0f} s "
      f"({ev.median() / 60:.1f} min) | maior parada = {ev.max()} s ({ev.max() / 60:.1f} min)")

ev1 = d1[f1 == 1]
ev2 = d2[f2 == 1]
print(f"\nPor carregamento (segundos por evento): carregamento 1 -> média {ev1.mean():.0f}, "
      f"mediana {ev1.median():.0f}, máx {ev1.max()} | carregamento 2 -> média {ev2.mean():.0f}, "
      f"mediana {ev2.median():.0f}, máx {ev2.max()}")

total_s = int(dur.sum())
print(f"\nImpacto total no período: {total_s} s = {total_s / 60:,.0f} min = {total_s / 3600:,.1f} h "
      f"em {int(flag.sum())} corridas")

ordenado = ev.sort_values(ascending=False).to_numpy()
top10 = ordenado[:max(1, round(len(ordenado) * 0.10))].sum() / ordenado.sum()
print(f"Concentração: os 10% de eventos mais longos respondem por {top10:.0%} do tempo parado")

dias = (camadas['dt_hora_consumo'].max() - camadas['dt_hora_consumo'].min()).days
meses = dias / 30.44
min_mes = total_s / 60 / meses
print(f"\nPeríodo coberto: {camadas['dt_hora_consumo'].min():%d/%m/%Y} a "
      f"{camadas['dt_hora_consumo'].max():%d/%m/%Y} ({meses:.1f} meses)")
print(f"Tempo parado por mês: ≈ {min_mes:,.0f} min ({min_mes / 60:,.1f} h) | "
      f"corridas por mês: ≈ {len(resp) / meses:,.0f} | eventos por mês: ≈ {flag.sum() / meses:,.0f}")

print("\nCenários: evitar uma fração dos eventos de carga alta (proporcional ao tempo parado):")
for frac in (0.10, 0.25, 0.50):
    linha = f"  evitar {frac:.0%} dos eventos -> ≈ {min_mes * frac:,.0f} min/mês poupados"
    if CUSTO_POR_MINUTO is not None:
        linha += f" (≈ R$ {min_mes * frac * CUSTO_POR_MINUTO:,.0f}/mês)"
    print(linha)
if CUSTO_POR_MINUTO is None:
    print("  (defina CUSTO_POR_MINUTO no topo do script para ver os valores em R$)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.6, 1]})
axes[0].hist(ev, bins=40, color=COR_COM, alpha=0.85)
axes[0].axvline(ev.median(), color='black', ls='--', lw=1.3)
axes[0].axvline(ev.mean(), color='black', ls=':', lw=1.3)
ymax = axes[0].get_ylim()[1]
axes[0].text(ev.median(), ymax * 0.97, f" mediana {ev.median():.0f} s", ha='left', va='top')
axes[0].text(ev.mean(), ymax * 0.88, f" média {ev.mean():.0f} s", ha='left', va='top')
axes[0].set_xlabel('Duração da parada (s)')
axes[0].set_ylabel('Nº de corridas')
axes[0].set_title(f'Duração das paradas por carga alta (n = {len(ev)})')
estilo(axes[0])

longo = pd.concat([pd.DataFrame({'carregamento': 1, 'duracao_s': ev1.values}),
                   pd.DataFrame({'carregamento': 2, 'duracao_s': ev2.values})])
sns.boxplot(data=longo, x='carregamento', y='duracao_s', hue='carregamento',
            palette=CORES_CARREG, legend=False, width=0.5, fliersize=3, ax=axes[1])
axes[1].set_xlabel('Carregamento')
axes[1].set_ylabel('Duração da parada (s)')
axes[1].set_title('Duração por carregamento')
estilo(axes[1])
plt.tight_layout()
plt.savefig('fig_5_3_duracao_paradas.png', dpi=150)
plt.show()


# ============================================================
# 6.0 IMPORTS NOVOS, CARREGAMENTO E PREPARO (em memória; nada é regravado)
# ============================================================

camadas = pd.read_csv('infos_camadas.csv', dtype={'cd_codigo_material': str}).drop_duplicates()
materiais = pd.read_csv('materiais_classificados.csv', dtype={'material': str})
resposta = pd.read_csv('VariavelResposta.csv')

camadas['dt_hora_consumo'] = pd.to_datetime(camadas['dt_hora_consumo'])
camadas['nu_carregamento'] = camadas['nu_carregamento'].astype(int)
camadas['nu_camada'] = camadas['nu_camada'].astype(int)
camadas['peso_t'] = camadas['qt_peso_carregamento'] / 1000                    # kg -> t

props = materiais.set_index('material')
camadas['classe'] = camadas['cd_codigo_material'].map(props['classe']).fillna('fora_dicionario')
camadas['densidade'] = camadas['cd_codigo_material'].map(props['densidade'])
camadas['volume_m3'] = camadas['peso_t'] / camadas['densidade']               # volume = peso / densidade

solidos = camadas[camadas['nu_carregamento'].isin([1, 2])].copy()             # só os cestões sólidos
sem_dens = solidos[solidos['densidade'].isnull()]
print(f"Camadas sólidas: {len(solidos)} | sem densidade no dicionário: {len(sem_dens)} "
      f"({sem_dens['cd_codigo_material'].unique().tolist()}, {sem_dens['peso_t'].sum():.1f} t no total)")
print("Essas camadas ficam FORA do volume estimado (peso registrado em peso_sem_densidade_t).")

FEATURES = ['volume_estimado_m3', 'peso_total_t', 'densidade_aparente', 'num_camadas', 'num_materiais',
            'camadas_peso_zero', 'peso_leve_t', 'peso_misto_t', 'peso_pesado_t',
            'prop_leve', 'prop_misto', 'prop_pesado', 'leve_no_meio_pct', 'pesado_fora_meio_pct']
FEATURES_BIN = ['base_leve', 'topo_leve', 'regra_ok', 'tem_gusa_liquido']
FEATURES_GRAFICO = ['volume_estimado_m3', 'peso_total_t', 'densidade_aparente',
                    'prop_leve', 'num_camadas', 'num_materiais']
FEATURES_TODAS = FEATURES + FEATURES_BIN
CARREGAMENTOS = [1, 2]


def fmt_p(p):
    return '<0.0001' if p < 0.0001 else f'{p:.4f}'


# ============================================================
# 6.1 AGREGANDO AS CAMADAS POR CESTÃO E POR CORRIDA
# ============================================================
titulo("6.1 FEATURES AGREGADAS (uma linha por cestão)")

cestoes = (solidos.groupby(CHAVE)
           .agg(peso_total_t=('peso_t', 'sum'),
                num_camadas=('nu_camada', 'nunique'),
                num_materiais=('cd_codigo_material', 'nunique'),
                camadas_peso_zero=('peso_t', lambda s: int((s == 0).sum())),
                volume_estimado_m3=('volume_m3', 'sum'),
                inicio_cestao=('dt_hora_consumo', 'min'))
           .reset_index())

peso_classe = (solidos.pivot_table(index=CHAVE, columns='classe', values='peso_t',
                                   aggfunc='sum', fill_value=0)
               .reindex(columns=['leve', 'misto', 'pesado'], fill_value=0)
               .add_prefix('peso_').add_suffix('_t'))
peso_sem_dens = (sem_dens.groupby(CHAVE)['peso_t'].sum().rename('peso_sem_densidade_t'))
cestoes = cestoes.set_index(CHAVE).join([peso_classe, peso_sem_dens]).reset_index()
cestoes['peso_sem_densidade_t'] = cestoes['peso_sem_densidade_t'].fillna(0)

for classe in ['leve', 'misto', 'pesado']:
    cestoes[f'prop_{classe}'] = cestoes[f'peso_{classe}_t'] / cestoes['peso_total_t']
# densidade aparente do cestão (t/m3): peso com densidade conhecida / volume estimado
cestoes['densidade_aparente'] = ((cestoes['peso_total_t'] - cestoes['peso_sem_densidade_t'])
                                 / cestoes['volume_estimado_m3'])

# --- posição das camadas (regra leve - pesado - leve, vista na 4.4) ---
solidos['n_camadas_cestao'] = solidos.groupby(CHAVE)['nu_camada'].transform('max')
solidos['pos_relativa'] = solidos['nu_camada'] / solidos['n_camadas_cestao']
solidos['zona'] = pd.cut(solidos['pos_relativa'], bins=[0, 0.3, 0.7, 1.0],
                         labels=['fundo', 'meio', 'topo'], include_lowest=True)
base_leve = (solidos[solidos['nu_camada'] == 1].groupby(CHAVE)['classe']
             .apply(lambda s: bool((s == 'leve').any())).rename('base_leve'))
topo_leve = (solidos[solidos['nu_camada'] == solidos['n_camadas_cestao']].groupby(CHAVE)['classe']
             .apply(lambda s: bool((s == 'leve').any())).rename('topo_leve'))
leve_no_meio = (solidos[solidos['classe'] == 'leve'].groupby(CHAVE)['zona']
                .apply(lambda s: float((s == 'meio').mean())).rename('leve_no_meio_pct'))
pesado_fora_meio = (solidos[solidos['classe'] == 'pesado'].groupby(CHAVE)['zona']
                    .apply(lambda s: float((s != 'meio').mean())).rename('pesado_fora_meio_pct'))
cestoes = (cestoes.set_index(CHAVE)
           .join([base_leve, topo_leve, leve_no_meio, pesado_fora_meio]).reset_index())
cestoes[['base_leve', 'topo_leve']] = cestoes[['base_leve', 'topo_leve']].fillna(False).astype(bool)
cestoes['regra_ok'] = cestoes['base_leve'] & cestoes['topo_leve']

# --- gusa líquido (carregamento 3) ---
# ATENÇÃO: o gusa líquido entra DEPOIS dos cestões. Só use tem_gusa_liquido como feature do modelo
# se a decisão de adicionar gusa já for conhecida antes de despejar o cestão (senão é vazamento).
gusa = (camadas[camadas['nu_carregamento'] == 3].groupby('cd_corrida')
        .agg(peso_carreg3_t=('peso_t', 'sum')).reset_index())
cestoes = cestoes.merge(gusa, on='cd_corrida', how='left')
cestoes['tem_gusa_liquido'] = cestoes['peso_carreg3_t'].notnull()
cestoes['peso_carreg3_t'] = cestoes['peso_carreg3_t'].fillna(0)

# --- variável resposta (um indicador por cestão + o indicador geral da corrida) ---
r_long = pd.concat([
    pd.DataFrame({'cd_corrida': resposta['cd_corrida'], 'nu_carregamento': 1,
                  'carga_alta': resposta['is_carga_alta_carregamento_1'],
                  'parada_s': resposta['qt_segundos_duracao_parada_carregamento_1']}),
    pd.DataFrame({'cd_corrida': resposta['cd_corrida'], 'nu_carregamento': 2,
                  'carga_alta': resposta['is_carga_alta_carregamento_2'],
                  'parada_s': resposta['qt_segundos_duracao_parada_carregamento_2']}),
])
cestoes = cestoes.merge(r_long, on=CHAVE, how='left')
cestoes = cestoes.merge(resposta[['cd_corrida', 'is_carga_alta']], on='cd_corrida', how='left')
cestoes[FEATURES_BIN] = cestoes[FEATURES_BIN].astype(int)

print(f"Cestões: {len(cestoes)} ({(cestoes['nu_carregamento'] == 1).sum()} no carregamento 1, "
      f"{(cestoes['nu_carregamento'] == 2).sum()} no 2) | sem resposta: {cestoes['carga_alta'].isnull().sum()}")
print(f"Taxa de carga alta por cestão: carregamento 1 = "
      f"{cestoes.loc[cestoes['nu_carregamento'] == 1, 'carga_alta'].mean():.1%} | carregamento 2 = "
      f"{cestoes.loc[cestoes['nu_carregamento'] == 2, 'carga_alta'].mean():.1%}")
print("\nResumo das features por carregamento:")
print(cestoes.groupby('nu_carregamento')[FEATURES[:9]].agg(['mean', 'std']).round(2).T.to_string())

# --- visão por corrida (uma linha por corrida, como pede o guia) ---
cols_corrida = ['peso_total_t', 'volume_estimado_m3', 'num_camadas', 'prop_leve', 'densidade_aparente']
largo = cestoes.pivot(index='cd_corrida', columns='nu_carregamento', values=cols_corrida)
largo.columns = [f"{c}_c{n}" for c, n in largo.columns]
corrida = (largo.reset_index()
           .merge(resposta, on='cd_corrida', how='left')
           .merge(gusa, on='cd_corrida', how='left'))
corrida['peso_carreg3_t'] = corrida['peso_carreg3_t'].fillna(0)
corrida['tem_gusa_liquido'] = corrida['peso_carreg3_t'] > 0
corrida['volume_total_m3'] = corrida[['volume_estimado_m3_c1', 'volume_estimado_m3_c2']].sum(axis=1)
corrida['peso_total_solidos_t'] = corrida[['peso_total_t_c1', 'peso_total_t_c2']].sum(axis=1)
print(f"\nVisão por corrida: {corrida.shape[0]} linhas x {corrida.shape[1]} colunas")

# ============================================================
# 6.2 COMPARAÇÃO: CARGA ALTA x NORMAL
# ============================================================
titulo("6.2 COMPARAÇÃO DAS FEATURES: carga alta x normal (por carregamento)")


def compara(df, features):
    """Média por grupo, diferença %, teste de Mann-Whitney e AUC univariada.
    AUC = chance de um cestão COM carga alta ter valor maior que um SEM (0,5 = feature não separa;
    1,0 = separa perfeitamente; abaixo de 0,5 = a relação é invertida)."""
    linhas = []
    for f in features:
        normal = df.loc[df['carga_alta'] == 0, f].dropna().astype(float)
        alta = df.loc[df['carga_alta'] == 1, f].dropna().astype(float)
        u, p = mannwhitneyu(alta, normal, alternative='two-sided')
        linhas.append({'feature': f, 'media_normal': normal.mean(), 'media_carga_alta': alta.mean(),
                       'dif_pct': (alta.mean() / normal.mean() - 1) * 100 if normal.mean() else np.nan,
                       'auc': u / (len(normal) * len(alta)), 'p': p})
    out = pd.DataFrame(linhas)
    out['forca'] = (out['auc'] - 0.5).abs()
    return out.sort_values('forca', ascending=False).reset_index(drop=True)


limite_bonf = 0.05 / len(FEATURES_TODAS)
comparacao = {}
for c in CARREGAMENTOS:
    d = cestoes[cestoes['nu_carregamento'] == c]
    comparacao[c] = compara(d, FEATURES_TODAS)
    tab = comparacao[c].assign(p_valor=comparacao[c]['p'].map(fmt_p),
                               signif=np.where(comparacao[c]['p'] < limite_bonf, '*', ''))
    print(f"\nCarregamento {c} ({int(d['carga_alta'].sum())} com carga alta de {len(d)} cestões) - "
          f"ordenado pela força da separação |AUC - 0,5|:")
    print(tab[['feature', 'media_normal', 'media_carga_alta', 'dif_pct', 'auc', 'p_valor', 'signif']]
          .round(3).to_string(index=False))
print(f"\n* = significativo mesmo após correção de Bonferroni (p < {limite_bonf:.4f}, {len(FEATURES_TODAS)} testes)")

print("\nVisão por corrida (como no guia), separando corridas COM e SEM gusa líquido:")
print(corrida.groupby(['tem_gusa_liquido', 'is_carga_alta'])
      [['volume_total_m3', 'peso_total_solidos_t']]
      .agg(['mean', 'count']).round(1).to_string())

fig, axes = plt.subplots(2, len(FEATURES_GRAFICO), figsize=(20, 8))
for i, c in enumerate(CARREGAMENTOS):
    d = cestoes[cestoes['nu_carregamento'] == c]
    for j, f in enumerate(FEATURES_GRAFICO):
        ax = axes[i, j]
        sns.boxplot(data=d, x='carga_alta', y=f, hue='carga_alta', palette={0: COR_SEM, 1: COR_COM},
                    legend=False, width=0.5, fliersize=2, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal', 'Carga alta'])
        ax.set_xlabel('')
        ax.set_ylabel(f'Carregamento {c}' if j == 0 else '')
        ax.set_title(f if i == 0 else '')
        estilo(ax)
plt.tight_layout()
plt.savefig('fig_6_2_boxplots_features.png', dpi=150)
plt.show()

# ============================================================
# 6.3 VOLUME ESTIMADO DO CESTÃO
# ============================================================
titulo("6.3 VOLUME ESTIMADO (peso / densidade)")


def sobreposicao(d, col, bins=30):
    """Coeficiente de sobreposição entre as duas distribuições (0 = separadas, 1 = idênticas)."""
    faixas = np.histogram_bin_edges(d[col].dropna(), bins=bins)
    p0, _ = np.histogram(d.loc[d['carga_alta'] == 0, col].dropna(), bins=faixas)
    p1, _ = np.histogram(d.loc[d['carga_alta'] == 1, col].dropna(), bins=faixas)
    return float(np.minimum(p0 / p0.sum(), p1 / p1.sum()).sum())


def avalia_limiar(d, col='volume_estimado_m3', semente=42):
    """Classificador de UMA regra (volume >= limiar). O limiar é escolhido em 70% dos cestões
    (maximizando a média entre recall e especificidade) e avaliado nos 30% restantes."""
    rng = np.random.default_rng(semente)
    idx = rng.permutation(len(d))
    corte = int(len(d) * 0.7)
    treino, teste = d.iloc[idx[:corte]], d.iloc[idx[corte:]]
    melhor_t, melhor_bal = None, -1
    for t in np.unique(np.quantile(treino[col], np.linspace(0.02, 0.98, 97))):
        pred = treino[col] >= t
        rec = pred[treino['carga_alta'] == 1].mean()
        esp = (~pred[treino['carga_alta'] == 0]).mean()
        if (rec + esp) / 2 > melhor_bal:
            melhor_bal, melhor_t = (rec + esp) / 2, t
    pred = teste[col] >= melhor_t
    y = teste['carga_alta'] == 1
    return {'limiar_m3': melhor_t, 'n_teste': len(teste), 'acuracia': (pred == y).mean(),
            'baseline_sempre_normal': 1 - y.mean(), 'recall': pred[y].mean(),
            'especificidade': (~pred[~y]).mean()}


for c in CARREGAMENTOS:
    d = cestoes[cestoes['nu_carregamento'] == c]
    linha_vol = comparacao[c].set_index('feature').loc['volume_estimado_m3']
    linha_peso = comparacao[c].set_index('feature').loc['peso_total_t']
    res = avalia_limiar(d)
    print(f"\nCarregamento {c}:")
    print(f"  volume médio: normal = {linha_vol['media_normal']:.1f} m3 | carga alta = "
          f"{linha_vol['media_carga_alta']:.1f} m3 ({linha_vol['dif_pct']:+.1f}%)")
    print(f"  AUC do volume = {linha_vol['auc']:.3f} | AUC do peso total = {linha_peso['auc']:.3f} "
          f"(0,5 = não separa)")
    print(f"  sobreposição das distribuições de volume: {sobreposicao(d, 'volume_estimado_m3'):.0%}")
    print(f"  modelo de 1 regra (volume >= {res['limiar_m3']:.1f} m3), testado em {res['n_teste']} cestões "
          f"não usados para escolher o limiar: acurácia = {res['acuracia']:.1%} "
          f"(sempre prever 'normal' = {res['baseline_sempre_normal']:.1%}) | "
          f"recall = {res['recall']:.1%} | especificidade = {res['especificidade']:.1%}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, c in zip(axes, CARREGAMENTOS):
    d = cestoes[cestoes['nu_carregamento'] == c]
    faixas = np.histogram_bin_edges(d['volume_estimado_m3'].dropna(), bins=30)
    for classe, cor, rotulo in [(0, COR_SEM, 'Normal'), (1, COR_COM, 'Carga alta')]:
        ax.hist(d.loc[d['carga_alta'] == classe, 'volume_estimado_m3'], bins=faixas, density=True,
                alpha=0.55, color=cor, label=f"{rotulo} (n = {(d['carga_alta'] == classe).sum()})")
    ax.set_xlabel('Volume estimado do cestão (m³)')
    ax.set_ylabel('Densidade')
    ax.set_title(f'Carregamento {c}')
    ax.legend()
    estilo(ax)
plt.tight_layout()
plt.savefig('fig_6_3_volume_por_classe.png', dpi=150)
plt.show()

# taxa de carga alta por faixa de volume (decis): existe um limiar?
fig, ax = plt.subplots(figsize=(9, 5))
for c in CARREGAMENTOS:
    d = cestoes[cestoes['nu_carregamento'] == c].copy()
    d['faixa'] = pd.qcut(d['volume_estimado_m3'], 10, duplicates='drop')
    t = d.groupby('faixa', observed=True).agg(volume=('volume_estimado_m3', 'mean'),
                                              taxa=('carga_alta', 'mean'), n=('carga_alta', 'size'))
    print(f"\nTaxa de carga alta por decil de volume - carregamento {c}:")
    print(t.assign(taxa=lambda x: (x['taxa'] * 100).round(1), volume=lambda x: x['volume'].round(1))
          .to_string())
    ax.plot(t['volume'], t['taxa'] * 100, marker='o', color=CORES_CARREG[c], label=f'Carregamento {c}')
ax.set_xlabel('Volume estimado médio do decil (m³)')
ax.set_ylabel('% de cestões com carga alta')
ax.set_title('Taxa de carga alta por faixa de volume (decis)')
ax.legend()
estilo(ax)
plt.tight_layout()
plt.savefig('fig_6_3b_taxa_por_faixa_volume.png', dpi=150)
plt.show()

# ============================================================
# 7.1 EXTRAINDO INFORMAÇÕES TEMPORAIS
# ============================================================
titulo("7.1 FEATURES TEMPORAIS (horário do início da montagem do cestão)")

t = cestoes['inicio_cestao']
cestoes['hora'] = t.dt.hour
cestoes['dia_semana'] = t.dt.dayofweek                       # 0 = segunda ... 6 = domingo
cestoes['dia_nome'] = cestoes['dia_semana'].map({0: 'seg', 1: 'ter', 2: 'qua', 3: 'qui',
                                                 4: 'sex', 5: 'sáb', 6: 'dom'})
cestoes['tipo_dia'] = cestoes['dia_semana'].map(
    lambda d: np.nan if pd.isnull(d) else ('fim de semana' if d >= 5 else 'dia útil'))
cestoes['mes'] = t.dt.strftime('%Y-%m')
cestoes['turno'] = np.select([(cestoes['hora'] >= 6) & (cestoes['hora'] < 14),
                              (cestoes['hora'] >= 14) & (cestoes['hora'] < 22)],
                             ['A (06-14h)', 'B (14-22h)'], default='C (22-06h)')
cestoes.loc[t.isnull(), 'turno'] = np.nan                    # sem horário: fica fora das análises temporais
cestoes['dias_desde_inicio'] = (t - t.min()).dt.total_seconds() / 86400

print(f"Cestões sem nenhum horário registrado: {int(t.isnull().sum())} (ficam de fora desta seção)")
print(f"Período: {t.min():%d/%m/%Y} a {t.max():%d/%m/%Y} | cestões por turno:")
print(cestoes['turno'].value_counts().sort_index().to_string())
print("Turnos (confirmar com o orientador): A = 06h-14h | B = 14h-22h | C = 22h-06h")
print("Cestões por mês (agosto/2025 e agosto/2026 são meses incompletos):")
print(cestoes['mes'].value_counts().sort_index().to_string())

# ============================================================
# 7.2 CARGA ALTA POR TURNO, DIA DA SEMANA E AO LONGO DO TEMPO
# ============================================================
titulo("7.2 TAXA DE CARGA ALTA POR TURNO, DIA DA SEMANA E MÊS")

MIN_N_MES = 30                                               # meses com menos cestões não entram na linha


def wilson(k, n, z=1.96):
    """Intervalo de confiança de 95% (Wilson) para uma proporção."""
    p = k / n
    centro = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    margem = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centro - margem, centro + margem


def taxa_por(df, col):
    g = (df.dropna(subset=[col]).groupby(['nu_carregamento', col])['carga_alta']
           .agg(k='sum', n='size').reset_index())
    g['taxa'] = g['k'] / g['n'] * 100
    g[['ic_inf', 'ic_sup']] = np.array([wilson(k, n) for k, n in zip(g['k'], g['n'])]) * 100
    return g


def barras_taxa(ax, g, col, ordem, titulo_ax):
    x, h = np.arange(len(ordem)), 0.38
    for desloc, c in ((-h / 2, 1), (h / 2, 2)):
        d = g[g['nu_carregamento'] == c].set_index(col).reindex(ordem)
        ax.bar(x + desloc, d['taxa'], width=h, color=CORES_CARREG[c], label=f'Carregamento {c}')
        ax.errorbar(x + desloc, d['taxa'], yerr=[d['taxa'] - d['ic_inf'], d['ic_sup'] - d['taxa']],
                    fmt='none', ecolor='black', elinewidth=1, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(ordem)
    ax.set_ylabel('% de cestões com carga alta')
    ax.set_title(titulo_ax)
    ax.legend()
    estilo(ax)


def testa_grupos(col):
    for c in CARREGAMENTOS:
        d = cestoes[cestoes['nu_carregamento'] == c].dropna(subset=[col])
        tab = pd.crosstab(d[col], d['carga_alta'])
        _, p, _, _ = chi2_contingency(tab)
        taxas = (d.groupby(col)['carga_alta'].mean() * 100).round(1).to_dict()
        print(f"  carregamento {c} | taxas (%): {taxas} | qui-quadrado p = {fmt_p(p)}")


ordem_turno = ['A (06-14h)', 'B (14-22h)', 'C (22-06h)']
ordem_dia = ['seg', 'ter', 'qua', 'qui', 'sex', 'sáb', 'dom']
g_turno, g_dia = taxa_por(cestoes, 'turno'), taxa_por(cestoes, 'dia_nome')
g_mes = taxa_por(cestoes, 'mes')

print("Por turno:")
testa_grupos('turno')
print("\nPor dia da semana:")
testa_grupos('dia_nome')
print("\nDia útil x fim de semana:")
testa_grupos('tipo_dia')

print("\nTendência no tempo (correlação ponto-bisserial entre a data e a carga alta):")
for c in CARREGAMENTOS:
    d = cestoes[cestoes['nu_carregamento'] == c].dropna(subset=['dias_desde_inicio'])
    r, p = pointbiserialr(d['carga_alta'], d['dias_desde_inicio'])
    print(f"  carregamento {c}: r = {r:+.3f} | p = {fmt_p(p)}")

meses = sorted(cestoes['mes'].dropna().unique())
print(f"\nTaxa mensal (meses com {MIN_N_MES}+ cestões por carregamento):")
tab_mes = (g_mes[g_mes['n'] >= MIN_N_MES].pivot(index='mes', columns='nu_carregamento', values=['taxa', 'n'])
           .round(1))
print(tab_mes.to_string())

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2)
barras_taxa(fig.add_subplot(gs[0, 0]), g_turno, 'turno', ordem_turno, 'Taxa de carga alta por turno')
barras_taxa(fig.add_subplot(gs[0, 1]), g_dia, 'dia_nome', ordem_dia, 'Taxa de carga alta por dia da semana')
ax = fig.add_subplot(gs[1, :])
for c in CARREGAMENTOS:
    d = (g_mes[(g_mes['nu_carregamento'] == c) & (g_mes['n'] >= MIN_N_MES)]
         .assign(x=lambda z: z['mes'].map({m: i for i, m in enumerate(meses)})).sort_values('x'))
    ax.plot(d['x'], d['taxa'], marker='o', color=CORES_CARREG[c], label=f'Carregamento {c}')
    ax.fill_between(d['x'], d['ic_inf'], d['ic_sup'], color=CORES_CARREG[c], alpha=0.15)
ax.set_xticks(range(len(meses)))
ax.set_xticklabels(meses, rotation=45, ha='right')
ax.set_ylabel('% de cestões com carga alta')
ax.set_title(f'Taxa de carga alta por mês (faixa = IC 95%; meses com menos de {MIN_N_MES} cestões omitidos)')
ax.legend()
estilo(ax)
plt.tight_layout()
plt.savefig('fig_7_2_padroes_temporais.png', dpi=150)
plt.show()

# ============================================================
# 8.1 MAPA DE CORRELAÇÕES
# ============================================================
titulo("8.1 CORRELAÇÕES (Spearman) ENTRE FEATURES E COM A CARGA ALTA")

for c in CARREGAMENTOS:
    d = cestoes[cestoes['nu_carregamento'] == c]
    corr = d[FEATURES_TODAS + ['carga_alta']].corr(method='spearman')
    ranking = corr['carga_alta'].drop('carga_alta')
    print(f"\nCarregamento {c} - correlação de cada feature com carga_alta (ordenado por |rho|):")
    print(ranking.reindex(ranking.abs().sort_values(ascending=False).index).round(3).to_string())

    pares = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURES_TODAS) for b in FEATURES_TODAS[i + 1:]
             if abs(corr.loc[a, b]) >= 0.8]
    print(f"\nCarregamento {c} - pares de features muito correlacionados entre si (|rho| >= 0,8): "
          f"{len(pares)}")
    for a, b, r in sorted(pares, key=lambda z: -abs(z[2]))[:12]:
        print(f"  {a} x {b}: {r:+.2f}")

    plt.figure(figsize=(13, 11))
    sns.heatmap(corr, annot=True, fmt='.2f', annot_kws={'size': 7}, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlação de Spearman'})
    plt.title(f'Mapa de correlações - carregamento {c}')
    plt.tight_layout()
    plt.savefig(f'fig_8_1_correlacoes_carreg{c}.png', dpi=150)
    plt.show()

# ============================================================
# 8.2 DISTRIBUIÇÕES POR CLASSE
# ============================================================
titulo("8.2 DISTRIBUIÇÕES DAS 6 FEATURES COM MAIOR SEPARAÇÃO (AUC)")

for c in CARREGAMENTOS:
    d = cestoes[cestoes['nu_carregamento'] == c]
    top = [f for f in comparacao[c]['feature'] if f in FEATURES][:6]
    print(f"Carregamento {c}: {top}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, f in zip(axes.flat, top):
        faixas = np.histogram_bin_edges(d[f].dropna(), bins=30)
        for classe, cor, rotulo in [(0, COR_SEM, 'Normal'), (1, COR_COM, 'Carga alta')]:
            ax.hist(d.loc[d['carga_alta'] == classe, f].dropna(), bins=faixas, density=True,
                    alpha=0.55, color=cor, label=rotulo)
        auc = comparacao[c].set_index('feature').loc[f, 'auc']
        ax.set_title(f"{f} (AUC = {auc:.2f})")
        ax.set_ylabel('Densidade')
        ax.legend()
        estilo(ax)
    fig.suptitle(f'Distribuição por classe - carregamento {c}', y=1.0)
    plt.tight_layout()
    plt.savefig(f'fig_8_2_distribuicoes_carreg{c}.png', dpi=150)
    plt.show()

# ============================================================
# 8.3 ANÁLISE BIVARIADA COM A RESPOSTA
# ============================================================
titulo("8.3 CORRELAÇÃO PONTO-BISSERIAL E TABELAS CRUZADAS")

linhas = []
for c in CARREGAMENTOS:
    d = cestoes[cestoes['nu_carregamento'] == c]
    for f in FEATURES_TODAS:
        x = d[['carga_alta', f]].dropna()
        r, p = pointbiserialr(x['carga_alta'], x[f].astype(float))
        linhas.append({'carregamento': c, 'feature': f, 'corr': r, 'p': p, 'n': len(x)})
pb = pd.DataFrame(linhas)
pb['signif_5pct'] = pb['p'] < 0.05
pb['signif_bonferroni'] = pb['p'] < limite_bonf
for c in CARREGAMENTOS:
    t_c = pb[pb['carregamento'] == c].copy()
    t_c = t_c.reindex(t_c['corr'].abs().sort_values(ascending=False).index)
    print(f"\nCarregamento {c} (significativas a 5%: {int(t_c['signif_5pct'].sum())} de {len(t_c)}; "
          f"após Bonferroni: {int(t_c['signif_bonferroni'].sum())}):")
    print(t_c.assign(p=t_c['p'].map(fmt_p))[['feature', 'corr', 'p', 'signif_5pct', 'signif_bonferroni']]
          .round(3).to_string(index=False))

print("\nFeatures categóricas (taxa de carga alta em % e qui-quadrado):")
for col in ['turno', 'tipo_dia', 'tem_gusa_liquido', 'regra_ok']:
    print(f"\n{col}:")
    testa_grupos(col)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for i, c in enumerate(CARREGAMENTOS):
    d = cestoes[cestoes['nu_carregamento'] == c]
    top = [f for f in pb[(pb['carregamento'] == c) & pb['feature'].isin(FEATURES)]
           .assign(a=lambda z: z['corr'].abs()).sort_values('a', ascending=False)['feature']][:4]
    for j, f in enumerate(top):
        ax = axes[i, j]
        sns.violinplot(data=d, x='carga_alta', y=f, hue='carga_alta', palette={0: COR_SEM, 1: COR_COM},
                       inner='quartile', cut=0, legend=False, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Normal', 'Carga alta'])
        ax.set_xlabel('')
        ax.set_ylabel(f'Carregamento {c}' if j == 0 else '')
        ax.set_title(f)
        estilo(ax)
plt.tight_layout()
plt.savefig('fig_8_3_violinos_top_features.png', dpi=150)
plt.show()

# ============================================================
# SALVAR (arquivos novos; os de entrada não são tocados)
# ============================================================
cestoes.to_csv('dataset_modelagem.csv', index=False)
corrida.to_csv('dataset_modelagem_corrida.csv', index=False)
pd.concat([comparacao[c].assign(carregamento=c) for c in CARREGAMENTOS]).to_csv(
    'tabela_comparacao_features.csv', index=False)
pb.to_csv('tabela_pontobisserial.csv', index=False)
print(f"\nSalvos: dataset_modelagem.csv ({cestoes.shape[0]} x {cestoes.shape[1]}), "
      f"dataset_modelagem_corrida.csv ({corrida.shape[0]} x {corrida.shape[1]}), "
      "tabela_comparacao_features.csv, tabela_pontobisserial.csv")