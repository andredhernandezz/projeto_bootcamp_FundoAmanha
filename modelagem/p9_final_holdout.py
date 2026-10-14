"""P9 · Avaliação FINAL no holdout (180 corridas). Rode UMA vez, só quando o modelo estiver decidido.

Treina no dev inteiro com o modelo/feature set escolhidos (P4/P5) e avalia no holdout uma única vez.
Trava: cria 'holdout_usado.lock'; uma segunda execução é recusada (para não 'ajustar olhando o teste').
Uso:  DADOS_DIR=. python p9_final_holdout.py --confirmo"""
import json, os, sys
import numpy as np, pandas as pd
import config as C, avalia as A, p1_dados, p2_features, p5_modelos

LOCK = os.path.join(C.SAIDAS, 'holdout_usado.lock')
if '--confirmo' not in sys.argv:
    sys.exit('Recusado: passe --confirmo se o modelo está definitivamente decidido.')
if os.path.exists(LOCK):
    sys.exit('Recusado: o holdout já foi usado (' + LOCK + '). Usá-lo de novo invalida a avaliação.')

d, _ = p2_features.construir(p1_dados.carregar())
esc = json.load(open(os.path.join(C.SAIDAS, 'features_escolhidas.json')))
mod = json.load(open(os.path.join(C.SAIDAS, 'modelo_escolhido.json')))
partes = []
for c in (1, 2):
    tr, te = d[d.dev & (d.nu_carregamento == c)], d[~d.dev & (d.nu_carregamento == c)].copy()
    A.checa_features(esc[str(c)])
    m = p5_modelos.CANDIDATOS[mod[str(c)]]().fit(tr[esc[str(c)]], tr[C.ALVO])
    te['p'] = m.predict_proba(te[esc[str(c)]])[:, 1]
    partes.append(te)
    mt = A.metricas(te); lo, hi = A.bootstrap_por_corrida(te, 'PR_AUC')
    print(f'cestão {c}  {mod[str(c)]}: ' + ', '.join(f'{k}={v:.3f}' for k, v in mt.items()) + f'  PR_AUC IC95=[{lo:.3f},{hi:.3f}]')
H = pd.concat(partes)
print('geral (2 cestões juntos): ', {k: round(v, 3) for k, v in A.metricas(H).items()})
H[['cd_corrida', 'nu_carregamento', 't', C.ALVO, 'parada_s', 'p']].to_csv(os.path.join(C.SAIDAS, 'holdout_previsoes.csv'), index=False)
open(LOCK, 'w').write('holdout avaliado\n')
