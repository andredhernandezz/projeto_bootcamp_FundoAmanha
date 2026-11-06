"""Pipeline de inferência EVCOMX (Random Forest, carga alta) — executável em Docker.

Subcomandos:
  verify    reproduz o teste original e confere métricas (mesmo critério do verificar_modelos.py)
  predict   previsão em lote: CSV de features -> CSV com risco e alerta
  evaluate  recalcula métricas e business case a partir das previsões de teste e grava JSON

Exemplos:
  python pipeline.py verify
  python pipeline.py predict --cestao 1 --input data_in/entrada.csv --output data_out/previsoes_c1.csv
  python pipeline.py evaluate --output data_out/avaliacao.json
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('EVCOMX_RF_DATA_DIR', BASE / 'assets'))
MODELS = Path(os.environ.get('EVCOMX_RF_MODEL_DIR', BASE / 'assets'))


def load_cfg():
    return json.loads((DATA / 'config.json').read_text(encoding='utf-8'))


def load_model(c, cfg):
    model = joblib.load(MODELS / f'random_forest_c{c}.joblib')
    cols = cfg['cestoes'][str(c)]['features']
    if list(model.feature_names_in_) != cols:
        raise ValueError(f'Features do modelo do cestão {c} não batem com o config.json')
    return model, cols


def read_input(path):
    df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig')
    df.columns = [x.strip() for x in df.columns]
    if not 0 < len(df) <= 100_000:
        raise ValueError('Envie entre 1 e 100.000 registros.')
    return df


def predict_frame(df, c, cfg, model, cols):
    missing = [x for x in cols if x not in df.columns]
    if missing:
        raise ValueError('Faltam colunas obrigatórias: ' + ', '.join(missing))
    xx = df.loc[:, cols].copy()
    for col in cols:
        s = xx[col]
        if not pd.api.types.is_numeric_dtype(s):
            s = s.astype('string').str.replace(',', '.', regex=False)
        xx[col] = pd.to_numeric(s, errors='raise')
    xx = xx.replace([np.inf, -np.inf], np.nan)
    if c == 2:
        a = xx['carga_alta_c1_ctx']
        if a.isna().any() or not a.isin([0, 1]).all():
            raise ValueError('Cestão 2: carga_alta_c1_ctx deve ser o resultado observado (0/1) do cestão 1.')
    return model.predict_proba(xx)[:, 1]


def cmd_predict(a):
    cfg = load_cfg()
    c = a.cestao
    model, cols = load_model(c, cfg)
    df = read_input(a.input)
    if 'cestao' in df and not pd.to_numeric(df['cestao'], errors='coerce').eq(c).all():
        raise ValueError('O arquivo contém cestão diferente do selecionado em --cestao.')
    p = predict_frame(df, c, cfg, model, cols)
    th = cfg['cestoes'][str(c)]['limiar'] if a.threshold is None else a.threshold
    out = df[[x for x in ['cd_corrida', 'cestao', 't'] if x in df]].copy()
    out['risco_rf'] = np.round(p, 6)
    out['alerta'] = (p >= th).astype(int)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f'Cestão {c}: {len(out)} registros, {int(out.alerta.sum())} alertas (limiar {th:.4f}) -> {a.output}')


def confusion(d, th=None):
    y = d['carga_alta'].to_numpy(int)
    al = d['alerta_original'].to_numpy(int) if th is None else (d['p'].to_numpy(float) >= th).astype(int)
    tp = int(((al == 1) & (y == 1)).sum()); fp = int(((al == 1) & (y == 0)).sum())
    fn = int(((al == 0) & (y == 1)).sum()); tn = int(((al == 0) & (y == 0)).sum())
    cap = float(d.loc[(al == 1) & (y == 1), 'parada_s'].sum() / 60)
    return dict(n=len(d), positivos=int(y.sum()), vp=tp, fp=fp, fn=fn, vn=tn,
                recall=tp / max(tp + fn, 1), precisao=tp / max(tp + fp, 1), minutos_capturados=cap)


def cmd_evaluate(a):
    from sklearn.metrics import average_precision_score, roc_auc_score
    cfg = load_cfg()
    T = pd.read_csv(DATA / 'previsoes_teste_rf.csv')
    res = {}
    for c in (1, 2):
        d = T[T.cestao.eq(c)].copy()
        d['parada_s'] = pd.to_numeric(d['parada_s']).fillna(0)
        m = confusion(d)
        m['roc_auc'] = float(roc_auc_score(d.carga_alta, d.p))
        m['pr_auc'] = float(average_precision_score(d.carga_alta, d.p))
        m['limiar'] = cfg['cestoes'][str(c)]['limiar']
        m['ganho_liquido_min'] = m['minutos_capturados'] * a.eficacia - a.custo_fp * m['fp']
        m['ganho_liquido_reais'] = m['ganho_liquido_min'] * a.valor_min
        res[f'cestao_{c}'] = m
    res['premissas'] = dict(eficacia=a.eficacia, custo_fp_min=a.custo_fp, valor_min_reais=a.valor_min,
                            nota='Premissas ilustrativas; sem experimento causal.')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=2, ensure_ascii=False))


def cmd_verify(_):
    sys.exit(subprocess.call([sys.executable, str(BASE / 'verificar_modelos.py')]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('verify').set_defaults(fn=cmd_verify)
    p = sub.add_parser('predict')
    p.add_argument('--cestao', type=int, choices=[1, 2], required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threshold', type=float, default=None, help='limiar; padrão = config.json')
    p.set_defaults(fn=cmd_predict)
    e = sub.add_parser('evaluate')
    e.add_argument('--output', default='data_out/avaliacao.json')
    e.add_argument('--eficacia', type=float, default=1.0)
    e.add_argument('--custo-fp', dest='custo_fp', type=float, default=1.0)
    e.add_argument('--valor-min', dest='valor_min', type=float, default=2500.0)
    e.set_defaults(fn=cmd_evaluate)
    a = ap.parse_args()
    try:
        a.fn(a)
    except Exception as ex:
        print(f'ERRO: {ex}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
