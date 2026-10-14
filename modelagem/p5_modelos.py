"""P5 · Modelos. Pergunta: um modelo mais forte (XGBoost/CatBoost) e/ou ajustes valem a pena?

Decisões:
- Features = as escolhidas no P4 (por cestão).
- Grade PEQUENA por modelo (só ~3.5 mil linhas e ~16% de positivos: modelos grandes decoram o passado).
- Pesos de classe x sem pesos: testado no HGB. Para ranking, pesos pouco mudam; sem pesos as
  probabilidades já saem mais próximas da realidade (importa na calibração).
- Escolha do vencedor por PR-AUC (OOF agrupado). Como a escolha usa a própria validação, ela é levemente
  otimista; o número honesto vem do holdout (P9)."""
import json, os, pandas as pd
import config as C, avalia as A, modelos as M

CANDIDATOS = {
    'LR C=0.05': M.lr(0.05), 'LR C=0.5': M.lr(0.5),
    'HGB d2': M.hgb(2, 0.05, 200), 'HGB d3': M.hgb(3, 0.05, 200), 'HGB d3 sem pesos': M.hgb(3, 0.05, 200, balanced=False),
    'XGB d2': M.xgb(2, 300), 'XGB d3': M.xgb(3, 300), 'XGB d3 spw=3': M.xgb(3, 300, spw=3.0),
    'CatBoost d4': M.cat(4, 300), 'CatBoost d4 sem pesos': M.cat(4, 300, balanced=False),
}


def rodar(d):
    esc = json.load(open(os.path.join(C.SAIDAS, 'features_escolhidas.json')))
    D = d[d.dev]; linhas = []; melhores = {}
    for c in (1, 2):
        Dc = D[D.nu_carregamento == c]; cols = esc[str(c)]; res = {}
        for nome, fab in CANDIDATOS.items():
            res[nome] = A.oof_temporal(Dc, cols, fab)
        T = A.tabela(res); T.insert(0, 'cestao', c); linhas.append(T)
        melhor = T.sort_values('PR_AUC', ascending=False).iloc[0]['config']
        melhores[str(c)] = melhor
        res[melhor][['cd_corrida', 'nu_carregamento', 't', C.ALVO, 'parada_s', 'p', 'fold']].to_csv(
            os.path.join(C.SAIDAS, f'oof_cestao{c}.csv'), index=False)
    R = pd.concat(linhas); R.to_csv(os.path.join(C.SAIDAS, 'p5_modelos.csv'), index=False)
    json.dump(melhores, open(os.path.join(C.SAIDAS, 'modelo_escolhido.json'), 'w'), indent=1)
    return R, melhores
