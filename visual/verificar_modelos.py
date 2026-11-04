"""Verifica de forma independente que os Random Forests reproduzem o teste original.
Uso: python verificar_modelos.py
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parent/'assets'
CFG=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
T=pd.read_csv(ROOT/'previsoes_teste_rf.csv')
V=pd.read_csv(ROOT/'previsoes_validacao_rf.csv')
EX=pd.read_csv(ROOT/'exemplos_teste_rf.csv')
expected={1:{'n':180,'events':26,'tp':22,'fp':44,'fn':4,'tn':110,'roc':0.8219280719280719,'ap':0.41039738515422586},
          2:{'n':172,'events':31,'tp':25,'fp':30,'fn':6,'tn':111,'roc':0.8798901853122855,'ap':0.7767174742531702}}
for c in [1,2]:
    model=joblib.load(ROOT/f'random_forest_c{c}.joblib')
    cols=CFG['cestoes'][str(c)]['features']; d=T.loc[T.cestao.eq(c)].copy()
    inp=EX.loc[EX.cestao.eq(c)].set_index('cd_corrida').loc[d.cd_corrida]
    p=model.predict_proba(inp[cols])[:,1]
    err=np.abs(p-d['p'].to_numpy()).max()
    th=CFG['cestoes'][str(c)]['limiar'];al=(p>=th).astype(int); y=d.carga_alta.to_numpy(dtype=int)
    tp=int(((al==1)&(y==1)).sum());fp=int(((al==1)&(y==0)).sum());fn=int(((al==0)&(y==1)).sum());tn=int(((al==0)&(y==0)).sum())
    exp=expected[c];metrics={'n':len(d),'events':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'roc':roc_auc_score(y,p),'ap':average_precision_score(y,p)}
    assert err<1e-10,(c,err)
    assert np.array_equal(al,d.alerta_original.to_numpy()),'alertas nao batem'
    for k in ('n','events','tp','fp','fn','tn'):assert metrics[k]==exp[k],(c,k,metrics[k],exp[k])
    assert np.isclose(metrics['roc'],exp['roc'],atol=1e-12)
    assert np.isclose(metrics['ap'],exp['ap'],atol=1e-12)
    assert len(cols)==({1:56,2:59}[c])
    assert model.named_steps['model'].n_estimators==180
    v=V[V.cestao.eq(c)]
    print(f'Cestao {c}: {len(d)} testes, {tp} VP, {fp} FP, {fn} FN, {tn} VN, AUC={metrics["roc"]:.4f}, AP={metrics["ap"]:.4f}, validacao={len(v)}, erro maximo score={err:.2e} [OK]')
print('Todos os testes passaram.')
