"""Inferência (Aula IX · deploy): risco calibrado de carga alta por cestão.

Entrada: CSV com as colunas de features construídas por p1_dados/p2_features (ex.: dataset_features.csv)
         e a coluna nu_carregamento (1 ou 2). Para um cestão NOVO, monte a linha com as mesmas funções
         (composição do plano de carga + histórico dos cestões anteriores).
Saída: CSV com risco_calibrado e alerta (1 = está entre os q% de maior risco daquele cestão).

Uso:   python prever.py --entrada saidas/dataset_features.csv --saida previsoes.csv [--q 0.10] [--ultimos 200]
Obs.:  com o dataset de desenvolvimento, as previsões são EM AMOSTRA (só demonstração)."""
import argparse, json, os
import numpy as np, pandas as pd
from catboost import CatBoostClassifier

PASTA = 'modelo_final'


def carregar():
    cfg = json.load(open(os.path.join(PASTA, 'config.json')))
    mods = {}
    for c in cfg['cestoes']:
        m = CatBoostClassifier(); m.load_model(os.path.join(PASTA, f'modelo_cestao{c}.cbm')); mods[c] = m
    return cfg, mods


def prever(df, q=0.10, cfg=None, mods=None):
    if cfg is None: cfg, mods = carregar()
    chave = str(q)
    out = []
    for c, info in cfg['cestoes'].items():
        x = df[df.nu_carregamento == int(c)]
        if x.empty: continue
        faltam = [f for f in info['features'] if f not in x.columns]
        assert not faltam, f'faltam colunas: {faltam}'
        p = mods[c].predict_proba(x[info['features']])[:, 1]
        z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        cal = 1 / (1 + np.exp(-(info['platt_coef'] * z + info['platt_intercepto'])))
        r = x[[k for k in ('cd_corrida', 'nu_carregamento', 't') if k in x.columns]].copy()
        r['risco_calibrado'] = cal.round(3)
        r['alerta'] = (p >= info['cortes_prob_bruta'][chave]).astype(int)
        out.append(r)
    return pd.concat(out).sort_index()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--entrada', required=True); ap.add_argument('--saida', default='previsoes.csv')
    ap.add_argument('--q', type=float, default=0.10); ap.add_argument('--ultimos', type=int, default=0)
    a = ap.parse_args()
    df = pd.read_csv(a.entrada, parse_dates=['t'] if 't' in pd.read_csv(a.entrada, nrows=1).columns else None)
    if a.ultimos: df = df.tail(a.ultimos)
    r = prever(df, a.q); r.to_csv(a.saida, index=False)
    print(f'{len(r)} cestões | alertas: {int(r.alerta.sum())} ({r.alerta.mean():.0%}) | risco médio {r.risco_calibrado.mean():.3f}')
    print(r.tail(8).to_string(index=False))
