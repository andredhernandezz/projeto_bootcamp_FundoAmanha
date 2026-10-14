"""P4 · Enriquecimento. Pergunta: quais grupos de features melhoram o modelo de verdade?

Método: partindo de BASE+TEMPO, (1) mede o ganho de cada grupo isoladamente (Δ PR-AUC pareado, IC95
por bootstrap de corridas); (2) seleção forward gulosa: adiciona o grupo de maior ganho enquanto o IC95 do
ganho excluir zero (parcimônia: com ~16% de positivos, ganho dentro do ruído não paga a complexidade). A seleção usa só HGB para ser barata; o IC reflete ruído amostral mas NÃO a otimização da
seleção (por isso o resultado final só vale depois do holdout)."""
import json, os, pandas as pd
import config as C, avalia as A, modelos as M

GRUPOS = {1: ['historico', 'posicao', 'materiais'],
          2: ['desfecho_c1', 'feats_c1', 'historico', 'posicao', 'materiais']}


def rodar(d, grupos):
    D = d[d.dev]
    fab = M.hgb()
    linhas, escolhidas = [], {}
    for c in (1, 2):
        Dc = D[D.nu_carregamento == c]
        ref_cols = C.BASE + C.TEMPO
        ref = A.oof_temporal(Dc, ref_cols, fab)
        base_pr = A.metricas(ref)['PR_AUC']
        # (1) ganho isolado
        for g in GRUPOS[c]:
            o = A.oof_temporal(Dc, ref_cols + grupos[g], fab)
            dlt, (lo, hi) = A.delta_pareado(o, ref, n=200)
            linhas.append(dict(cestao=c, etapa='isolado', grupo=g, PR_AUC=A.metricas(o)['PR_AUC'],
                               delta=dlt, IC_lo=lo, IC_hi=hi))
        # (2) forward guloso
        atual, cols, restantes = ref, list(ref_cols), list(GRUPOS[c])
        while restantes:
            cand = {g: A.oof_temporal(Dc, cols + grupos[g], fab) for g in restantes}
            melhor = max(cand, key=lambda g: A.metricas(cand[g])['PR_AUC'])
            dlt, (lo, hi) = A.delta_pareado(cand[melhor], atual, n=200)
            linhas.append(dict(cestao=c, etapa='forward', grupo=melhor,
                               PR_AUC=A.metricas(cand[melhor])['PR_AUC'], delta=dlt, IC_lo=lo, IC_hi=hi))
            if lo <= 0:   # IC95 do ganho inclui zero: indistinguível de ruído -> para (parcimônia)
                break
            cols += grupos[melhor]; atual = cand[melhor]; restantes.remove(melhor)
        escolhidas[str(c)] = cols
    R = pd.DataFrame(linhas).round(4)
    R.to_csv(os.path.join(C.SAIDAS, 'p4_enriquecimento.csv'), index=False)
    json.dump(escolhidas, open(os.path.join(C.SAIDAS, 'features_escolhidas.json'), 'w'), indent=1)
    return R, escolhidas
