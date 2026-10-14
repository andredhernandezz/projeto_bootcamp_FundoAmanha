"""P1 · Dados e split.

Decisões:
- Linhas ordenadas por início do cestão (cronologia real).
- Dev = corridas do treino.csv; holdout = as demais (últimas 180 corridas). O holdout é só MARCADO aqui.
- Features de histórico usam apenas cestões ANTERIORES (shift) -> disponíveis na hora da decisão.
"""
import numpy as np, pandas as pd, os
import config as C


def carregar():
    d = pd.read_csv(os.path.join(C.DADOS, 'dataset_modelagem.csv'))
    tr = pd.read_csv(os.path.join(C.DADOS, 'treino.csv'))
    d['t'] = pd.to_datetime(d['inicio_cestao'])
    d = d.sort_values(['t', 'nu_carregamento']).reset_index(drop=True)
    d['dev'] = d.cd_corrida.isin(set(tr.cd_corrida))
    return d


def valida_split(d):
    dev, ho = d[d.dev], d[~d.dev]
    assert not set(dev.cd_corrida) & set(ho.cd_corrida)
    # O treino.csv é a divisão oficial. Pode haver sobreposição mínima na fronteira (ex.: uma corrida
    # do dev que começa depois da primeira do holdout); medimos e avisamos em vez de reordenar o split.
    sobrepostos = int((dev.t >= ho.t.min()).sum())
    return dict(cestoes_dev_apos_inicio_holdout=sobrepostos,
                corridas_dev=dev.cd_corrida.nunique(), corridas_holdout=ho.cd_corrida.nunique(),
                taxa_dev=dev[C.ALVO].mean(), taxa_holdout=ho[C.ALVO].mean())


def adiciona_historico(d):
    """Features de contexto recente. Tudo com shift(1): só olha cestões já concluídos.
    NÃO usamos o intervalo desde o cestão anterior: depende do horário de início do cestão atual,
    que ainda não é conhecido quando a decisão de como carregar é tomada."""
    d = d.copy()
    a = d[C.ALVO]
    d['h_prev_alta'] = a.shift(1)
    d['h_taxa10'] = a.shift(1).rolling(10, min_periods=3).mean()
    d['h_taxa50'] = a.shift(1).rolling(50, min_periods=10).mean()
    # cestões desde a última carga alta (contagem só com passado)
    idx = np.arange(len(d)); ult = pd.Series(np.where(a.shift(1) == 1, idx - 1, np.nan)).ffill()
    d['h_desde_ult_alta'] = (idx - ult).clip(upper=100)
    # do cestão 1 para o 2 da mesma corrida
    c1 = d[d.nu_carregamento == 1].set_index('cd_corrida')
    for novo, col in [('c1_carga_alta', C.ALVO), ('c1_peso_total_t', 'peso_total_t'),
                      ('c1_densidade', 'densidade_aparente'), ('c1_regra_ok', 'regra_ok'),
                      ('c1_prop_leve', 'prop_leve')]:
        d[novo] = np.where(d.nu_carregamento == 2, d.cd_corrida.map(c1[col]), np.nan)
    d['c1_carga_alta'] = d['c1_carga_alta'].fillna(0)  # no cestão 1 a coluna não é usada
    return d


if __name__ == '__main__':
    d = carregar(); print(valida_split(d))
