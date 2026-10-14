"""Roda o pipeline de desenvolvimento (P1-P7). NÃO usa o holdout.
Uso (na pasta com os CSVs da EDA):  DADOS_DIR=. python main.py"""
import json, os, sys, time
import pandas as pd
import config as C, p8_explicabilidade, p11_figuras, p1_dados, p2_features, p3_baseline, p4_enriquecimento, p5_modelos, p6_calibracao_limiar, p7_negocio

t0 = time.time()
d = p1_dados.carregar(); print('SPLIT', p1_dados.valida_split(d))
d, grupos = p2_features.construir(d)
d.to_csv(os.path.join(C.SAIDAS, 'dataset_features.csv'), index=False)
json.dump(grupos, open(os.path.join(C.SAIDAS, 'grupos_features.json'), 'w'), indent=1)

print('\n== P3 baseline =='); print(p3_baseline.rodar(d).to_string(index=False))
print('\n== P4 enriquecimento =='); R, esc = p4_enriquecimento.rodar(d, grupos); print(R.to_string(index=False))
for c in esc: print(f'cestão {c}: {len(esc[c])} features')

print('\n== P5 modelos =='); R5, melhores = p5_modelos.rodar(d); print(R5.to_string(index=False)); print('escolhidos:', melhores)
print('\n== P6 calibração e limiar =='); cal, melhor, rel, L = p6_calibracao_limiar.rodar()
print(cal.to_string(index=False)); print('calibrador escolhido:', melhor); print(rel); print(L.to_string(index=False))
print('\n== P7 negócio =='); info, melhor_q = p7_negocio.rodar(); print(info); print(melhor_q.to_string(index=False))
print('\n== P8 explicabilidade =='); [print(f'cestão {c}:', T.head(5).variavel.tolist()) for c, T in p8_explicabilidade.rodar(d).items()]
print('\n== P11 figuras =='); p11_figuras.rodar()
print(f'({time.time()-t0:.0f}s)')
