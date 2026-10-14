"""Validação temporal, métricas e comparação pareada. Usado por todos os experimentos."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import TimeSeriesSplit
import config as C


def checa_features(cols):
    """Trava de segurança: nenhuma feature proibida entra num modelo."""
    ruins = [c for c in cols if c in C.PROIBIDAS]
    assert not ruins, f'features proibidas (vazamento): {ruins}'


def oof_temporal(Dc, cols, fabrica, n_splits=C.N_SPLITS):
    """Previsões fora-da-amostra com validação cruzada temporal AGRUPADA POR CORRIDA.

    As corridas são ordenadas no tempo; cada fold treina no passado e testa no bloco seguinte.
    Retorna um DataFrame com as linhas que foram testadas e a coluna 'p'.
    """
    checa_features(cols)
    corridas = Dc.groupby('cd_corrida')['t'].min().sort_values().index.to_numpy()
    partes = []
    for k, (tri, tei) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(corridas)):
        a = Dc[Dc.cd_corrida.isin(set(corridas[tri]))]
        b = Dc[Dc.cd_corrida.isin(set(corridas[tei]))].copy()
        m = fabrica().fit(a[cols], a[C.ALVO])
        b['p'] = m.predict_proba(b[cols])[:, 1]
        b['fold'] = k
        partes.append(b)
    return pd.concat(partes)


def _m(y, p, par, q=C.FRACAO_ALERTA):
    k = max(1, int(round(len(y) * q)))
    top = np.argsort(-p, kind='stable')[:k]
    return dict(ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                prev=y.mean(), lift=y[top].mean() / y.mean(),
                recall=y[top].sum() / y.sum(), precisao=y[top].mean(),
                min_capt=par[top].sum() / par.sum())


def metricas(o, q=C.FRACAO_ALERTA):
    """Métricas do conjunto de previsões 'o' (colunas: carga_alta, parada_s, p).
    lift/recall/precisão/min_capt: sinalizando a fração q de maiores probabilidades."""
    return _m(o[C.ALVO].to_numpy(), o['p'].to_numpy(), o['parada_s'].to_numpy(), q)


def _boot(o, fn_arr, n, seed):
    """Reamostra CORRIDAS inteiras (cestões da mesma corrida são dependentes). fn_arr(y,p,par,pref)."""
    rng = np.random.default_rng(seed)
    cods, inv = np.unique(o['cd_corrida'].to_numpy(), return_inverse=True)
    linhas = [np.where(inv == i)[0] for i in range(len(cods))]
    y, p, par = o[C.ALVO].to_numpy(), o['p'].to_numpy(), o['parada_s'].to_numpy()
    pr = o['p_ref'].to_numpy() if 'p_ref' in o else None
    vals = []
    for _ in range(n):
        ix = np.concatenate([linhas[i] for i in rng.integers(0, len(cods), len(cods))])
        vals.append(fn_arr(y[ix], p[ix], par[ix], None if pr is None else pr[ix]))
    return np.percentile(vals, [2.5, 97.5])


def bootstrap_por_corrida(o, metrica='PR_AUC', n=300, seed=C.SEED):
    """IC 95% (percentis) de uma métrica, reamostrando corridas."""
    return _boot(o, lambda y, p, par, pr: _m(y, p, par)[metrica], n, seed)


def delta_pareado(o_novo, o_ref, metrica='PR_AUC', n=300, seed=C.SEED):
    """Diferença (novo - ref) na MESMA amostra, com IC95 por bootstrap de corridas.
    Comparar nos mesmos cestões é bem mais preciso que comparar dois ICs separados."""
    j = o_novo[['cd_corrida', C.ALVO, 'parada_s', 'p']].copy()
    j['p_ref'] = o_ref['p'].reindex(j.index).to_numpy()
    y, p, par, pr = (j[C.ALVO].to_numpy(), j['p'].to_numpy(), j['parada_s'].to_numpy(), j['p_ref'].to_numpy())
    est = _m(y, p, par)[metrica] - _m(y, pr, par)[metrica]
    ic = _boot(j, lambda y, p, par, pr: _m(y, p, par)[metrica] - _m(y, pr, par)[metrica], n, seed)
    return est, ic


def tabela(res):
    """res: dict nome -> oof. Retorna tabela de métricas com IC95 do PR-AUC."""
    linhas = []
    for nome, o in res.items():
        m = metricas(o); lo, hi = bootstrap_por_corrida(o, 'PR_AUC', n=200)
        linhas.append({'config': nome, **m, 'PR_AUC_IC': f'[{lo:.3f}, {hi:.3f}]'})
    return pd.DataFrame(linhas).round(3)
