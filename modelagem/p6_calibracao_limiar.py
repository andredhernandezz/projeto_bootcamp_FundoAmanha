"""P6 · Calibração e limiar de alerta.

Calibração: o modelo ordena bem, mas 'prob=0,40' precisa significar ~40%. Ajusta-se um calibrador
(Platt = regressão logística; isotônica) nas previsões OOF MAIS ANTIGAS e avalia nas MAIS RECENTES
(Brier e log-loss), respeitando o tempo. Limiar: tabela por fração de cestões sinalizados, juntando os dois
cestões, com a parada capturada em minutos."""
import json, os, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import config as C


def _logit(p): p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4); return np.log(p / (1 - p)).reshape(-1, 1)


def rodar():
    O = pd.concat([pd.read_csv(os.path.join(C.SAIDAS, f'oof_cestao{c}.csv'), parse_dates=['t']) for c in (1, 2)])
    O = O.sort_values(['t', 'nu_carregamento']).reset_index(drop=True)
    corte = len(O) // 2
    a, b = O.iloc[:corte], O.iloc[corte:]
    linhas = [dict(metodo='sem calibrar', brier=brier_score_loss(b[C.ALVO], b.p), logloss=log_loss(b[C.ALVO], b.p.clip(1e-4, 1-1e-4)))]
    platt = LogisticRegression(C=1e6).fit(_logit(a.p), a[C.ALVO])
    pp = platt.predict_proba(_logit(b.p))[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(a.p, a[C.ALVO]); pi = iso.predict(b.p)
    for nome, p in [('Platt', pp), ('isotônica', pi)]:
        linhas.append(dict(metodo=nome, brier=brier_score_loss(b[C.ALVO], p), logloss=log_loss(b[C.ALVO], np.clip(p, 1e-4, 1-1e-4))))
    cal = pd.DataFrame(linhas).round(4); cal.to_csv(os.path.join(C.SAIDAS, 'p6_calibracao.csv'), index=False)
    melhor = cal.iloc[1:].sort_values('logloss').iloc[0].metodo
    # tabela de confiabilidade (metade recente, probabilidade calibrada pelo método escolhido)
    p_cal = pp if melhor == 'Platt' else pi
    rel = pd.DataFrame({'p': p_cal, 'y': b[C.ALVO].to_numpy()})
    rel['faixa'] = pd.qcut(rel.p, 5, duplicates='drop')
    rel = rel.groupby('faixa', observed=True).agg(p_media=('p', 'mean'), taxa_real=('y', 'mean'), n=('y', 'size')).round(3)
    rel.to_csv(os.path.join(C.SAIDAS, 'p6_confiabilidade.csv'))
    # limiar por fração sinalizada (pool dos 2 cestões, ranking pela probabilidade bruta; ranking não muda com calibrador monotônico)
    linhas = []
    for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        k = int(round(len(O) * q)); top = O.nlargest(k, 'p')
        linhas.append(dict(fracao_alerta=q, n_alertas=k, limiar_prob_bruta=top.p.min(), recall=top[C.ALVO].sum() / O[C.ALVO].sum(),
                           precisao=top[C.ALVO].mean(), lift=top[C.ALVO].mean() / O[C.ALVO].mean(),
                           falsos_alarmes=int((top[C.ALVO] == 0).sum()), min_capturados=top.parada_s.sum() / 60,
                           pct_minutos=top.parada_s.sum() / O.parada_s.sum()))
    L = pd.DataFrame(linhas).round(3); L.to_csv(os.path.join(C.SAIDAS, 'p6_limiar.csv'), index=False)
    return cal, melhor, rel, L
