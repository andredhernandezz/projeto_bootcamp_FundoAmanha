"""Treino final (Aula IX · deploy). Gera os artefatos usados por prever.py na pasta 'modelo_final/'.

Uso:  DADOS_DIR=. python treinar_final.py                 (treina só com o desenvolvimento)
      DADOS_DIR=. python treinar_final.py --com-holdout   (SÓ depois da avaliação final: treina com tudo)

Artefatos: modelo_cestao{1,2}.cbm (CatBoost), config.json (features, calibrador Platt e cortes de alerta).
Calibrador e cortes saem das previsões fora-da-amostra do desenvolvimento (validação temporal)."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import config as C, avalia as A, p1_dados, p2_features, p5_modelos

OUT = 'modelo_final'
Q = [0.05, 0.10, 0.20]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p)).reshape(-1, 1)


def main(com_holdout=False):
    os.makedirs(OUT, exist_ok=True)
    d, _ = p2_features.construir(p1_dados.carregar())
    esc = json.load(open(os.path.join(C.SAIDAS, 'features_escolhidas.json')))
    cfg = {'treinado_com_holdout': com_holdout, 'cestoes': {}}
    fab = p5_modelos.CANDIDATOS['CatBoost d4']
    for c in (1, 2):
        cols = esc[str(c)]; A.checa_features(cols)
        Dc = d[(d.dev | com_holdout) & (d.nu_carregamento == c)]
        Dd = d[d.dev & (d.nu_carregamento == c)]
        oof = A.oof_temporal(Dd, cols, fab)                       # calibração e cortes SÓ com o dev
        platt = LogisticRegression(C=1e6).fit(logit(oof.p), oof[C.ALVO])
        cortes = {str(q): float(np.quantile(oof.p, 1 - q)) for q in Q}   # corte na prob. bruta
        m = fab().fit(Dc[cols], Dc[C.ALVO]); m.save_model(os.path.join(OUT, f'modelo_cestao{c}.cbm'))
        cfg['cestoes'][str(c)] = dict(features=cols, platt_coef=float(platt.coef_[0][0]),
                                      platt_intercepto=float(platt.intercept_[0]), cortes_prob_bruta=cortes,
                                      n_treino=int(len(Dc)))
        print(f'cestão {c}: treinado com {len(Dc)} cestões; cortes brutos {cortes}')
    json.dump(cfg, open(os.path.join(OUT, 'config.json'), 'w'), indent=1)


if __name__ == '__main__':
    main('--com-holdout' in sys.argv)
