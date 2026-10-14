"""P8 · Explicabilidade (Aula VIII). Só usa o desenvolvimento.

Três olhares complementares sobre o MESMO problema:
1. Importância por permutação (por fold temporal): quanto a PR-AUC CAI se embaralharmos uma variável
   no bloco de teste. Mede o que o modelo realmente USA para acertar fora da amostra.
2. SHAP do CatBoost (modelo treinado em todo o dev): contribuição média absoluta e direção do efeito.
3. Coeficientes padronizados da regressão logística: leitura linear simples, para comparar.
Atenção: importância não é causa. Mostra no que o modelo se apoia, não o que mudar no forno."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import Pool
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
import config as C, modelos as M, p1_dados, p2_features

LEG = {'h_taxa10': 'taxa de carga alta nos últimos 10 cestões', 'h_taxa50': 'taxa de carga alta nos últimos 50 cestões',
       'h_prev_alta': 'cestão anterior teve carga alta', 'h_desde_ult_alta': 'cestões desde a última carga alta',
       'c1_carga_alta': 'cestão 1 da corrida teve carga alta'}


def permutacao(Dc, cols, fab, n_rep=5, seed=C.SEED):
    rng = np.random.default_rng(seed)
    corridas = Dc.groupby('cd_corrida')['t'].min().sort_values().index.to_numpy()
    queda = {c: [] for c in cols}
    for tri, tei in TimeSeriesSplit(n_splits=C.N_SPLITS).split(corridas):
        a = Dc[Dc.cd_corrida.isin(set(corridas[tri]))]; b = Dc[Dc.cd_corrida.isin(set(corridas[tei]))]
        m = fab().fit(a[cols], a[C.ALVO])
        ref = average_precision_score(b[C.ALVO], m.predict_proba(b[cols])[:, 1])
        for c in cols:
            qs = []
            for _ in range(n_rep):
                x = b[cols].copy(); x[c] = rng.permutation(x[c].to_numpy())
                qs.append(ref - average_precision_score(b[C.ALVO], m.predict_proba(x)[:, 1]))
            queda[c].append(np.mean(qs))
    return pd.DataFrame({'variavel': cols, 'queda_PR_AUC': [np.mean(v) for v in queda.values()],
                         'dp_entre_folds': [np.std(v) for v in queda.values()]})


def shap_catboost(Dc, cols, fab):
    m = fab().fit(Dc[cols], Dc[C.ALVO])
    sv = m.get_feature_importance(Pool(Dc[cols], Dc[C.ALVO]), type='ShapValues')[:, :-1]
    X = Dc[cols].reset_index(drop=True)
    direcao = [np.corrcoef(sv[:, i], X[c].fillna(X[c].median()))[0, 1] if X[c].nunique() > 1 else 0 for i, c in enumerate(cols)]
    return pd.DataFrame({'variavel': cols, 'shap_medio_abs': np.abs(sv).mean(0), 'direcao_corr': np.round(direcao, 2)})


def coef_lr(Dc, cols):
    m = M.lr(0.5)().fit(Dc[cols], Dc[C.ALVO])
    return pd.DataFrame({'variavel': cols, 'coef_padronizado': m[-1].coef_[0]})


def rodar(d):
    esc = json.load(open(os.path.join(C.SAIDAS, 'features_escolhidas.json')))
    D = d[d.dev]; saidas = {}
    figdir = os.path.join(C.SAIDAS, 'figuras'); os.makedirs(figdir, exist_ok=True)
    for c in (1, 2):
        Dc = D[D.nu_carregamento == c]; cols = esc[str(c)]
        fab = __import__('p5_modelos').CANDIDATOS['CatBoost d4']
        P = permutacao(Dc, cols, fab); S = shap_catboost(Dc, cols, fab); L = coef_lr(Dc, cols)
        T = P.merge(S, on='variavel').merge(L, on='variavel').sort_values('queda_PR_AUC', ascending=False)
        T['descricao'] = T.variavel.map(LEG).fillna('')
        T.round(4).to_csv(os.path.join(C.SAIDAS, f'p8_explicabilidade_cestao{c}.csv'), index=False)
        saidas[c] = T
        top = T.head(10).iloc[::-1]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].barh(top.variavel, top.queda_PR_AUC, xerr=top.dp_entre_folds, color='#2a6f97'); ax[0].set_title(f'Cestão {c}: queda de PR-AUC ao embaralhar')
        t2 = T.sort_values('shap_medio_abs', ascending=False).head(10).iloc[::-1]
        ax[1].barh(t2.variavel, t2.shap_medio_abs, color='#c8553d'); ax[1].set_title(f'Cestão {c}: SHAP médio absoluto')
        for a in ax: a.spines[['top', 'right']].set_visible(False)
        plt.tight_layout(); plt.savefig(os.path.join(figdir, f'importancia_cestao{c}.png'), dpi=140); plt.close()
    return saidas


if __name__ == '__main__':
    d, _ = p2_features.construir(p1_dados.carregar())
    for c, T in rodar(d).items():
        print(f'\n== cestão {c} =='); print(T.head(12).round(4).to_string(index=False))
