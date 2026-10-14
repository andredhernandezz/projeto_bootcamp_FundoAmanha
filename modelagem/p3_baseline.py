"""P3 · Baseline. Pergunta: quanto sinal existe só com a composição das camadas?
Decisão: dois modelos simples (LR interpretável e boosting raso) com pesos de classe, um por cestão."""
import pandas as pd, os
import config as C, avalia as A, modelos as M


def rodar(d):
    D = d[d.dev]
    linhas = []
    for c in (1, 2):
        Dc = D[D.nu_carregamento == c]
        for nome, cols in {'composição': C.BASE, 'composição+tempo': C.BASE + C.TEMPO}.items():
            for mn, fab in {'LR': M.lr(), 'HGB': M.hgb()}.items():
                o = A.oof_temporal(Dc, cols, fab)
                linhas.append({'cestao': c, 'features': nome, 'modelo': mn, **A.metricas(o)})
    R = pd.DataFrame(linhas).round(3)
    R.to_csv(os.path.join(C.SAIDAS, 'p3_baseline.csv'), index=False)
    return R
