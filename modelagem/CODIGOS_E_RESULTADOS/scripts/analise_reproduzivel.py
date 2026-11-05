#!/usr/bin/env python3
"""EVCOMX: modelos de carga alta por carregamento, teste futuro intocado.
Uso: python analise_reproduzivel.py <diretorio_csv_original> <features_cestao.csv> <pasta_saida>
Os scripts entregues como artefato acompanham os arquivos de dados do usuario.
"""
import sys, json, warnings, os
from pathlib import Path
from datetime import date
import numpy as np, pandas as pd
import holidays
FEATURES_ONLY = '--features-only' in sys.argv[1:]
ARGS = [arg for arg in sys.argv[1:] if arg != '--features-only']
if len(ARGS) != 3:
    raise SystemExit(
        'Uso: analise_reproduzivel.py <diretorio_csv_original> '
        '<features_cestao.csv> <pasta_saida> [--features-only]'
    )
SRC,FN,OUT=map(Path,ARGS);OUT.mkdir(exist_ok=True,parents=True)
for p in ['graficos','tabelas','predicoes','testes']: (OUT/p).mkdir(exist_ok=True)
F=pd.read_csv(FN,parse_dates=['inicio_cestao','inicio_corrida'])
assert len(F)==3501 and F['cd_corrida'].nunique()==1799
# Relogio: CSV treino mostra +00:00 para dt_corrida. Tratar data bruta como UTC e converter a JF/Minas -03.
# Sensibilidade: datas SEM a conversao sao examinadas em testes, porque um sistema pode etiquetar UTC indevidamente.
FER=holidays.Brazil(subdiv='MG',years=[2025,2026]); FER_DATE=set(pd.Timestamp(k) for k in FER)
FAC=set(pd.Timestamp(z) for z in ['2025-12-24','2025-12-31','2026-02-16','2026-02-17','2026-02-18','2026-04-20','2026-06-05'])
# F possui datetimes naive que correspondem aos campos em UTC de treino.csv
local_time=F['inicio_cestao']-pd.Timedelta(hours=3)
F['data_local']=local_time.dt.normalize()
F['data_utc']=F['inicio_cestao'].dt.normalize()
F['feriado_local']=F['data_local'].isin(FER_DATE).astype(int)
F['feriado_utc']=F['data_utc'].isin(FER_DATE).astype(int)
F['facultativo_local']=F['data_local'].isin(FAC).astype(int)
F['fim_semana_local']=(local_time.dt.dayofweek>=5).astype(int)
F['dia_semana_local']=local_time.dt.dayofweek
F['hora_local']=local_time.dt.hour
F['turno_local']=pd.cut(F['hora_local'],[-1,5,13,21,24],labels=[0,1,2,0],ordered=False).astype(float)
F['sin_hora_local']=np.sin(2*np.pi*F['hora_local']/24)
F['cos_hora_local']=np.cos(2*np.pi*F['hora_local']/24)
F['mes_sin']=np.sin(2*np.pi*local_time.dt.month/12)
F['mes_cos']=np.cos(2*np.pi*local_time.dt.month/12)
F['feriado_vizinhanca']=F['data_local'].apply(lambda d: int(d-pd.Timedelta(days=1) in FER_DATE or d+pd.Timedelta(days=1) in FER_DATE))
# Quando timestamp faltante foi herdado de outra carga, não apresentá-lo como horário confiável do cestão.
for col in ['hora_local','dia_semana_local','sin_hora_local','cos_hora_local','mes_sin','mes_cos','feriado_local','fim_semana_local','feriado_vizinhanca','turno_local']:
    F.loc[F['horario_herdado']==1,col]=np.nan
# Features materiais derivadas da frequencia do TREINO apenas, nao do futuro
cam=pd.read_csv(SRC/'infos_camadas.csv',usecols=['cd_corrida','nu_carregamento','cd_codigo_material','qt_peso_carregamento'])
cam=cam[cam.nu_carregamento.isin([1,2])].copy()
keys=F[F.split=='treino'][['cd_corrida','nu_carregamento']]
cf=cam.merge(keys,on=['cd_corrida','nu_carregamento'],how='inner')
freq=(cf.groupby('cd_codigo_material')[['cd_corrida','nu_carregamento']]
      .apply(lambda x:len(x.drop_duplicates()),include_groups=False)/len(keys))
frequent=freq[freq>0.10].index.astype(str).tolist()
ct=cam[cam.cd_codigo_material.astype(str).isin(frequent)].copy()
ct['material']=ct.cd_codigo_material.astype(str)
weight=cam.groupby(['cd_corrida','nu_carregamento']).qt_peso_carregamento.sum()
props=ct.pivot_table(index=['cd_corrida','nu_carregamento'],columns='material',values='qt_peso_carregamento',aggfunc='sum',fill_value=0).div(weight,axis=0).fillna(0)
props.columns=['participacao_'+s for s in props.columns]
F=F.merge(props,left_on=['cd_corrida','nu_carregamento'],right_index=True,how='left')
matcols=list(props.columns)
F[matcols]=F[matcols].fillna(0)
core=['peso_total_t','volume_estimado_m3','densidade_media','n_camadas','prop_leve','prop_pesado']
composition=['peso_leve_t','peso_pesado_t','peso_misto_t','prop_misto','peso_sem_densidade_t','vol_leve_m3',
             'prop_vol_leve','rendimento_medio','energia_por_ton','n_materiais','n_baias','hhi_materiais']+matcols
ordering=['base_leve','topo_leve','base_pesado','topo_pesado','violacao_regra','prop_leve_topo','prop_pesado_base',
          'pos_media_pesado','peso_topo_30pct_t','peso_base_30pct_t','leve_no_meio_pct','pesado_fora_meio_pct']
context=['tempo_montagem_min','intervalo_medio_min','hora_local','sin_hora_local','cos_hora_local',
         'dia_semana_local','fim_semana_local','feriado_local','feriado_vizinhanca','mes_sin','mes_cos',
         'taxa_ca_ult5','taxa_ca_ult10','taxa_ca_ult20','gap_corrida_anterior_h','corridas_desde_ultima_ca']
interactions=['vol_x_prop_leve','peso_x_hhi','peso_leve_topo_t','pesado_x_densidade']
c2ctx=['carga_alta_c1_ctx','delta_peso_12','razao_peso_21','vol_acumulado_m3','delta_prop_pesado_12','jaccard_materiais_12','intervalo_c1_c2_min']

if FEATURES_ONLY:
    F.to_csv(OUT / 'features_modelo_rf.csv', index=False)
    print(f'Features RF geradas: {len(F)} linhas, {len(F.columns)} colunas.')
    raise SystemExit(0)

from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (roc_auc_score,average_precision_score,f1_score,fbeta_score,precision_score,
                             recall_score, confusion_matrix,brier_score_loss)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=FutureWarning)
rng = np.random.default_rng(20261020)

# No pos-evento: is_carga_alta, y_corrida, parada_s, gusa_liquido, volume da outra cesta para c1.
results=[]; selection={}; holdout=[]; perf_imp=[]; hist=[]; warnings_meta=[]

def metr(y,p,thr):
    z=(p>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel()
    return dict(n=len(y),eventos=int(sum(y)),roc_auc=round(roc_auc_score(y,p),4),pr_auc=round(average_precision_score(y,p),4),
                precision=round(precision_score(y,z,zero_division=0),4),recall=round(recall_score(y,z,zero_division=0),4),
                f2=round(fbeta_score(y,z,beta=2,zero_division=0),4),f1=round(f1_score(y,z,zero_division=0),4),
                brier=round(brier_score_loss(y,p),4),taxa_alertas=round(z.mean(),4),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))

def cut_budget(y,p,budget=.35):
    uniq=np.unique(np.r_[1.0000001,p]); opts=[]
    for v in uniq:
        alert=(p>=v).mean()
        if alert<=budget+0.00000001:
            opts.append((fbeta_score(y,p>=v,beta=2,zero_division=0), -alert, float(v)))
    return max(opts)[2] if opts else 1.0000001

def build(kind,cols):
    if kind in ['Logistica L2','Logistica L1','Logistica fisica','Logistica filtro','Logistica RFE']:
        pen='l1' if kind=='Logistica L1' else 'l2'
        return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=False)),('scale',StandardScaler()),
                         ('model',LogisticRegression(C=0.25 if pen=='l1' else 1.0,penalty=pen,solver='liblinear',max_iter=1500,random_state=42))])
    if kind=='Random Forest':
        return Pipeline([('imputer',SimpleImputer(strategy='median')),
             ('model',RandomForestClassifier(n_estimators=180,min_samples_leaf=12,max_features=.65,random_state=42,n_jobs=-1))])
    if kind=='Gradient Boosting':
        return Pipeline([('imputer',SimpleImputer(strategy='median')),
             ('model',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=9,min_samples_leaf=25,l2_regularization=2,learning_rate=.045,random_state=42))])
    if kind=='XGBoost':
        return Pipeline([('imputer',SimpleImputer(strategy='median')),
             ('model',XGBClassifier(n_estimators=160,max_depth=2,learning_rate=.04,subsample=.85,
                                   colsample_bytree=.9,min_child_weight=12,reg_lambda=8,objective='binary:logistic',
                                   eval_metric='logloss',random_state=42,n_jobs=2))])
    raise ValueError(kind)

for c in [1,2]:
    data=F[F.nu_carregamento==c].copy()
    data=data.sort_values(['inicio_corrida','cd_corrida'])
    extra=c2ctx if c==2 else []
    allcols=list(dict.fromkeys(core+composition+ordering+context+interactions+extra))
    allcols=[v for v in allcols if data[v].nunique(dropna=True)>1]
    for vv in allcols:
        data[vv]=pd.to_numeric(data[vv],errors='coerce').replace([np.inf,-np.inf],np.nan)
    tr=data[data.split=='treino'];va=data[data.split=='valid'];te=data[data.split=='teste']
    ytr=tr.y.astype(int).values; yva=va.y.astype(int).values; yte=te.y.astype(int).values
    # Mann-Whitney features numeric and Holm/BH correction, fit on TRAIN only
    mw=[]
    for col in allcols:
        a=tr.loc[tr.y==0,col].dropna();b=tr.loc[tr.y==1,col].dropna()
        if min(len(a),len(b))<10 or min(a.nunique(),b.nunique())<2:continue
        u,pv=stats.mannwhitneyu(a,b,alternative='two-sided'); auc=u/len(a)/len(b)
        mw.append((col,pv,auc,a.mean(),b.mean(),len(a)+len(b)))
    mdf=pd.DataFrame(mw,columns=['variavel','p_bruto','auc_normal_maior','media_y0','media_y1','n']).sort_values('p_bruto')
    mdf['p_BH']=multipletests(mdf.p_bruto,method='fdr_bh')[1]
    mdf['p_Holm']=multipletests(mdf.p_bruto,method='holm')[1]
    mdf['efeito_auc']=abs(mdf.auc_normal_maior-.5)+.5
    mdf.to_csv(OUT/'testes'/f'mann_whitney_treino_c{c}.csv',index=False)
    # filtro nao supervisionado de redundancia: para par com rho>0.9 preservar variavel que aparece antes na lista
    cor=tr[allcols].corr(method='spearman').abs()
    kept=[];drop=[]
    for f in allcols:
        high=[q for q in kept if pd.notna(cor.loc[f,q]) and cor.loc[f,q]>.90]
        if high:drop.append((f,high[0],round(cor.loc[f,high[0]],3)))
        else:kept.append(f)
    pd.DataFrame(drop,columns=['variavel_redundante','mantida','rho_abs']).to_csv(OUT/'testes'/f'redundancia_spearman_c{c}.csv',index=False)
    filt=[f for f in kept if f in set(mdf[mdf.p_BH<.05].variavel)]
    # acrescentar nucleo fisico caso filtro estrito elimine alguma parte do signal
    filt=list(dict.fromkeys([f for f in core if f in kept]+filt))
    selection[f'c{c}']={'candidatas':allcols,'frequentes_treino':frequent,'filtro_BH':filt,'removidas_redundancia':drop}
    # wrapper RFE em treino, jamais teste
    xi=SimpleImputer(strategy='median').fit_transform(tr[kept]); xs=StandardScaler().fit_transform(xi)
    nfeat=min(14,len(kept))
    rfe=RFE(LogisticRegression(solver='liblinear',max_iter=900,random_state=42),n_features_to_select=nfeat,step=.2)
    rfe.fit(xs,ytr)
    sel_rfe=[a for a,on in zip(kept,rfe.support_) if on]
    selection[f'c{c}']['RFE']=sel_rfe
    # L1 selection as embedded
    l1=build('Logistica L1',kept).fit(tr[kept],ytr)
    sel_l1=[f for f,v in zip(kept,l1.named_steps['model'].coef_[0]) if abs(v)>1e-7]
    selection[f'c{c}']['L1']=sel_l1
    # Modelos e subsets
    configs=[('Logistica fisica',core),('Logistica L2',kept),('Logistica L1',kept),
             ('Logistica filtro',filt),('Logistica RFE',sel_rfe),('Random Forest',kept),
             ('Gradient Boosting',kept),('XGBoost',kept)]
    models={}
    for name,cols in configs:
        mod=build(name,cols); mod.fit(tr[cols],ytr)
        pv=mod.predict_proba(va[cols])[:,1]
        th=cut_budget(yva,pv)
        r=metr(yva,pv,th);r.update(carregamento=c,modelo=name,split='validacao',limiar=th,n_features=len(cols))
        results.append(r);models[name]=(mod,cols,th)
    # Selecionar maior f2 respeitando budget e desempatar pr_auc, recall, menor nr features
    valid=[r for r in results if r['carregamento']==c and r['split']=='validacao']
    selected=max(valid,key=lambda r:(r['f2'],r['pr_auc'],r['recall'],-r['n_features']))
    name=selected['modelo'];model,cols,th=models[name]
    selection[f'c{c}']['modelo_campeao']=name
    selection[f'c{c}']['features_modelo_campeao']=cols
    selection[f'c{c}']['limiar']=th
    # testar UMA VEZ a escolha feita na validacao, sem recalibrar no futuro
    ph=model.predict_proba(te[cols])[:,1];m=metr(yte,ph,th)
    m.update(carregamento=c,modelo=name,split='teste',limiar=th,n_features=len(cols));holdout.append(m)
    target=te[['cd_corrida','nu_carregamento','inicio_corrida','parada_s_cestao','y']].copy()
    target['risco_previsto']=ph;target['alerta']=(ph>=th).astype(int)
    target.to_csv(OUT/'predicoes'/f'predicoes_teste_c{c}.csv',index=False)
    # importancias permutation VALIDACAO AUC PR, interpretacao preditiva
    pin=permutation_importance(model,va[cols],yva,scoring='average_precision',random_state=21,n_repeats=4,n_jobs=1)
    pi=pd.DataFrame(dict(feature=cols,importancia_media=pin.importances_mean,desvio=pin.importances_std)).sort_values('importancia_media',ascending=False)
    pi.to_csv(OUT/'tabelas'/f'importancia_permutacao_validacao_c{c}.csv',index=False)
    perf_imp.extend([dict(carregamento=c,**r) for r in pi.head(15).to_dict(orient='records')])
    fig,ax=plt.subplots(figsize=(8,5.0));q=pi.head(12).iloc[::-1]
    ax.barh(q.feature,q.importancia_media,color='#2A668C');ax.axvline(0,ls=':',c='black');ax.set_xlabel('Queda no PR-AUC na permutacao (validacao)')
    ax.set_title(f'C{c}: importancia por permutacao - {name}');fig.tight_layout();fig.savefig(OUT/'graficos'/f'importancias_c{c}.png',dpi=160);plt.close(fig)
    # tempo poupavel em teoria: soma duracoes dos TP (teto), nao poupanca causal
    detected=target[(target.y==1)&(target.alerta==1)].parada_s_cestao.sum()/60
    allmin=target[target.y==1].parada_s_cestao.sum()/60
    selection[f'c{c}']['min_eventos_detectados_teto_nao_causal']=round(float(detected),1)
    selection[f'c{c}']['min_eventos_total_teste']=round(float(allmin),1)
    # Ablacao mesmo modelo/hiperparametros e limiar otimizado na propria validacao
    ablation={'completo':cols,'sem_feriados':[q for q in cols if not q.startswith('feriado')],
              'sem_ordem':[q for q in cols if q not in ordering],
              'sem_composicao':[q for q in cols if q not in composition]}
    for variant,cc in ablation.items():
        if not cc:continue
        md=build(name,cc).fit(tr[cc],ytr);vv=md.predict_proba(va[cc])[:,1]
        aa=metr(yva,vv,cut_budget(yva,vv)); hist.append(dict(carregamento=c,modelo=name,conjunto=variant,n_features=len(cc),f2_validacao=aa['f2'],pr_auc_validacao=aa['pr_auc'],recall_validacao=aa['recall']))
    # barplot validacao
    rs=pd.DataFrame(valid).sort_values('f2',ascending=False)
    fig,ax=plt.subplots(figsize=(9,4.7));ax.barh(rs.modelo[::-1],rs.f2[::-1],color='#2A668C')
    ax.set_xlim(0,1);ax.set_xlabel('F2 na validacao, alertas <= 35%');ax.set_title(f'C{c}: comparacao de modelos');fig.tight_layout();fig.savefig(OUT/'graficos'/f'modelos_c{c}.png',dpi=160);plt.close(fig)
    print('C',c,'selecionado',name,'valid f2',selected['f2'],'teste',m)

pd.DataFrame(results+holdout).to_csv(OUT/'tabelas'/'comparacao_modelos.csv',index=False)
pd.DataFrame(hist).to_csv(OUT/'tabelas'/'ablacoes_validacao.csv',index=False)
with open(OUT/'selecao_e_parametros.json','w') as fo:json.dump(selection,fo,indent=2,ensure_ascii=False,default=str)

# FERiados: estudo observacional - 1 linha/corrida, margem de erro por dias (cluster), nao por corridas.
R=F[F.nu_carregamento==1].copy();R['y_corrida']=R.y_corrida.astype(int)
# para feriado usar a data da primeira camada sem atribuir a corrida com horario herdado: c1 tem hora confiavel majoritariamente
R=R[R.horario_herdado==0].copy()
R['mes']=R.data_local.dt.to_period('M').astype(str)
R['semana']=R.data_local.dt.dayofweek
rt=R.groupby('feriado_local').y_corrida.agg(['sum','size','mean']);print('FERIADOS GLOBAL',rt.to_string())
a=R[R.feriado_local==1].y_corrida;b=R[R.feriado_local==0].y_corrida
ct=pd.crosstab(R.feriado_local,R.y_corrida).reindex(index=[0,1],columns=[0,1],fill_value=0)
OR,p_fisher=stats.fisher_exact(ct.values)
rs=[]
def fer_test(k,dd,cal='feriado_local'):
    aa=dd[dd[cal]==1].y_corrida;bb=dd[dd[cal]==0].y_corrida
    table=pd.crosstab(dd[cal],dd.y_corrida).reindex(index=[0,1],columns=[0,1],fill_value=0)
    o,p=stats.fisher_exact(table.values)
    return dict(teste=k,n_feriado=len(aa),eventos_feriado=int(aa.sum()),taxa_feriado=float(aa.mean()) if len(aa) else None,
                n_outro=len(bb),eventos_outro=int(bb.sum()),taxa_outro=float(bb.mean()) if len(bb) else None,
                dif_pp=100*(aa.mean()-bb.mean()) if len(aa) and len(bb) else None,odds_ratio=float(o),p_fisher=float(p))
rs.append(fer_test('calendario_local_corrida',R))
R['feriado_utc']=R.feriado_utc.astype(int)
rs.append(fer_test('calendario_utc_sem_conversao',R,'feriado_utc'))
rs.append(fer_test('somente_dias_uteis',R[R.fim_semana_local==0]))
# permutacao por dia com estratos MES + dia util/fim de semana, preservando numero de feriados em cada estrato
D=R.groupby('data_local',as_index=False).agg(n=('y_corrida','size'),s=('y_corrida','sum'),fer=('feriado_local','first'))
D['mes']=D.data_local.dt.to_period('M').astype(str);D['fds']=(D.data_local.dt.dayofweek>=5).astype(int)
obs=D.loc[D.fer==1,'s'].sum()/D.loc[D.fer==1,'n'].sum()-D.loc[D.fer==0,'s'].sum()/D.loc[D.fer==0,'n'].sum()
strata=D.groupby(['mes','fds']).indices
nperm=1500;perms=[]
for _ in range(nperm):
    perm=np.zeros(len(D),dtype=bool)
    for indices in strata.values():
        kk=int(D.iloc[indices].fer.sum())
        if kk:perm[rng.choice(indices,size=kk,replace=False)]=True
    perms.append(D.loc[perm,'s'].sum()/D.loc[perm,'n'].sum()-D.loc[~perm,'s'].sum()/D.loc[~perm,'n'].sum())
perms=np.array(perms);p_perm=(1+sum(abs(perms)>=abs(obs)))/(nperm+1)
# bootstrap cluster dia dentro de feriado/não feriado
boot=[]
for _ in range(1200):
    A=D[D.fer==1].sample(frac=1,replace=True,random_state=int(rng.integers(2**31-1)))
    B=D[D.fer==0].sample(frac=1,replace=True,random_state=int(rng.integers(2**31-1)))
    boot.append(A.s.sum()/A.n.sum()-B.s.sum()/B.n.sum())
ci=np.quantile(boot,[.025,.975])
rs.append(dict(teste='permutacao_por_dia_estratificada_mes_fds',n_feriado=len(D[D.fer==1]),eventos_feriado=None,
               taxa_feriado=None,n_outro=len(D[D.fer==0]),eventos_outro=None,taxa_outro=None,dif_pp=100*obs,odds_ratio=None,p_fisher=float(p_perm)))
# GLM binomial dia/feriado com IC robusto por dia, ajuste calendario mensal, weekday, tendencia
# colinearidade moderada e poucos feriados: controlar ano/mes (factor mes) e fim de semana
try:
    X=pd.get_dummies(R[['feriado_local','fim_semana_local','mes','dia_semana_local']],columns=['mes'],drop_first=True,dtype=float)
    X=X.drop(columns=['dia_semana_local']);X=sm.add_constant(X).astype(float)
    fit=sm.GLM(R.y_corrida,X,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':R.data_local})
    coef=fit.params['feriado_local'];se=fit.bse['feriado_local'];p_adj=fit.pvalues['feriado_local']
    adj={'or_ajustado':float(np.exp(coef)),'ic95_or':[float(np.exp(coef-1.96*se)),float(np.exp(coef+1.96*se))],
         'p_ajustado':float(p_adj),'metodo':'GLM binomial; efeitos fixos de mes; fim semana; erros cluster por dia'}
except Exception as e:adj={'erro':str(e)}
# documento individual por dia e tabela de taxa
Fdays=R[R.feriado_local==1].groupby('data_local').agg(corridas=('y_corrida','size'),eventos=('y_corrida','sum'),taxa=('y_corrida','mean')).reset_index()
Fdays['nome']=Fdays.data_local.dt.date.map(FER)
Fdays.to_csv(OUT/'testes'/'feriados_por_dia.csv',index=False)
pd.DataFrame(rs).to_csv(OUT/'testes'/'teste_hipotese_feriados.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(10.5,4))
axs[0].bar(['Feriado','Outro dia'],[a.mean()*100,b.mean()*100],color=['#D87A32','#246B91'])
axs[0].set_ylabel('Corridas com carga alta (%)');axs[0].set_title(f'Feriados de Juiz de Fora: Fisher p={p_fisher:.3f}')
for i,v in enumerate([a,b]):axs[0].text(i,100*v.mean()+1,f'{100*v.mean():.1f}% (n={len(v)})',ha='center')
axs[1].hist(perms*100,bins=38,color='#82A9C0');axs[1].axvline(obs*100,color='#C05023',lw=2,label=f'{obs*100:+.1f} p.p.')
axs[1].set_title(f'Permutacao em dias (p={p_perm:.3f})');axs[1].set_xlabel('Diferenca simulada, pontos percentuais');axs[1].legend()
for ax in axs:
    ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(OUT/'graficos'/'feriados_teste.png',dpi=165);plt.close(fig)
fer_out={'p_fisher':p_fisher,'or_fisher':OR,'p_permutacao_dia_estratificada':p_perm,'efeito_pp':100*obs,
          'ci_cluster_pp':[round(float(100*x),2) for x in ci],'n_dias_feriado':int(D.fer.sum()),
          'n_corridas_feriado':int(len(a)),'n_corridas_outros':int(len(b)),
          'n_feriados_holdout':int(F[(F.nu_carregamento==1)&(F.split=='teste')].feriado_local.sum()),
          'ajuste':adj}
with open(OUT/'testes'/'resultado_feriados.json','w') as fo:json.dump(fer_out,fo,ensure_ascii=False,indent=2)
# EDA e qualidade com alertas sobre target e denominadores
rdata=pd.read_csv(SRC/'VariavelResposta.csv'); absent=set(rdata.cd_corrida)-set(F[F.nu_carregamento==2].cd_corrida)
qual={'n_corridas':len(rdata),'n_cestoes':len(F),'n_c1':sum(F.nu_carregamento==1),'n_c2':sum(F.nu_carregamento==2),
      'c2_ausentes':len(absent),'ausentes_c2_com_alvo_positivo':int(rdata[rdata.cd_corrida.isin(absent)].is_carga_alta_carregamento_2.sum()),
      'eventos_corrida':int(rdata.is_carga_alta.sum()),'tempo_parado_total_min':round(rdata.qt_segundos_parada_carga_alta.sum()/60,1),
      'eventos_c1':int(rdata.is_carga_alta_carregamento_1.sum()),'eventos_c2_registrados':int(F[F.nu_carregamento==2].y.sum()),
      'n_train_c1':int(len(F[(F.nu_carregamento==1)&(F.split=='treino')])),
      'n_val_c1':int(len(F[(F.nu_carregamento==1)&(F.split=='valid')])),
      'n_test_c1':int(len(F[(F.nu_carregamento==1)&(F.split=='teste')])),
      'n_train_c2':int(len(F[(F.nu_carregamento==2)&(F.split=='treino')])),
      'n_val_c2':int(len(F[(F.nu_carregamento==2)&(F.split=='valid')])),
      'n_test_c2':int(len(F[(F.nu_carregamento==2)&(F.split=='teste')]))}
with open(OUT/'auditoria_dados.json','w') as fo:json.dump(qual,fo,indent=2,ensure_ascii=False)
# export features/auditoria com metadata e alvo separado
F.to_csv(OUT/'tabelas'/'features_consolidadas_auditaveis.csv',index=False)
print('AUDITORIA',qual)
print('FERIADOS',fer_out)
print(pd.DataFrame(results+holdout).to_string(index=False))
