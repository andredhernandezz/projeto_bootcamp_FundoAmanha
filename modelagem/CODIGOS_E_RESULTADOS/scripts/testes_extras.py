import pandas as pd,numpy as np,json,sys
from pathlib import Path
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path('resultado_final');F=pd.read_csv(p/'tabelas/features_consolidadas_auditaveis.csv',parse_dates=['inicio_corrida','data_local'])
L=[]
# paired loads with c2 existing
both=F.pivot(index='cd_corrida',columns='nu_carregamento',values='y').dropna()
n00=int(((both[1]==0)&(both[2]==0)).sum()); n01=int(((both[1]==0)&(both[2]==1)).sum());n10=int(((both[1]==1)&(both[2]==0)).sum());n11=int(((both[1]==1)&(both[2]==1)).sum())
x=mcnemar([[n00,n01],[n10,n11]],exact=True)
L.append(dict(test='McNemar pareado c1-c2',H0='Probabilidade c1 = probabilidade c2 na mesma corrida',n=len(both),estatistica=f'disc={n01} c2 / {n10} c1',p=float(x.pvalue),efeito=f'c1={both[1].mean():.3f};c2={both[2].mean():.3f}'))
y=stats.fisher_exact([[n00,n01],[n10,n11]])
L.append(dict(test='Fisher dependencia c1-c2',H0='Independencia dos resultados no mesmo ciclo',n=len(both),estatistica=f'OR={y.statistic:.2f}',p=float(y.pvalue),efeito=f'c2 se c1=1:{both[both[1]==1][2].mean():.3f};se c1=0:{both[both[1]==0][2].mean():.3f}'))
for c in [1,2]:
 g=F[(F.nu_carregamento==c)&(F.split=='treino')].copy()
 g['fim_semana']=g.data_local.dt.dayofweek>=5
 for v in ['prop_pesado','volume_estimado_m3','peso_total_t','hhi_materiais']:
  a=g[g.y==1][v].dropna();b=g[g.y==0][v].dropna();t,pv=stats.mannwhitneyu(a,b,alternative='two-sided')
  L.append(dict(test=f'Mann-Whitney {v} c{c}',H0='Mesma distribuicao com/sem evento no treino',n=len(a)+len(b),estatistica=f'U={t:.0f}',p=float(pv),efeito=f'auc positivo={t/(len(a)*len(b)):.3f}, media evento={a.mean():.2f} x sem={b.mean():.2f}'))
 tab=pd.crosstab(g['fim_semana'],g.y).reindex(index=[False,True],columns=[0,1],fill_value=0)
 chi,pv,_,_=stats.chi2_contingency(tab)
 L.append(dict(test=f'Qui quadrado semana c{c}',H0='Independencia fim semana e evento',n=len(g),estatistica=f'chi2={chi:.2f}',p=float(pv),efeito=f'taxa semana={g[g.fim_semana==False].y.mean():.3f}, fim semana={g[g.fim_semana==True].y.mean():.3f}'))
 # concept drift no holdout (teste apenas descricao e pos-hoc)
 q=F[(F.nu_carregamento==c)&(F.split=='teste')]
 z=F[(F.nu_carregamento==c)&(F.split=='treino')]
 for v in ['prop_pesado','volume_estimado_m3']:
  stat,pv=stats.ks_2samp(z[v].dropna(),q[v].dropna())
  L.append(dict(test=f'KS drift treino vs teste {v} c{c}',H0='Distribuicao covariavel estavel',n=len(z)+len(q),estatistica=f'D={stat:.3f}',p=float(pv),efeito=f'media treino={z[v].mean():.2f}; teste={q[v].mean():.2f}'))
# Wilson test CIs
main=pd.read_csv(p/'tabelas/comparacao_modelos.csv');w=[]
for _,r in main[main.split=='teste'].iterrows():
 rc=proportion_confint(r.tp,r.tp+r.fn,method='wilson'); pc=proportion_confint(r.tp,r.tp+r.fp,method='wilson')
 w.append(dict(carregamento=int(r.carregamento),recall=float(r.recall),recall_ic95_low=rc[0],recall_ic95_high=rc[1],precisao=float(r.precision),precision_ic95_low=pc[0],precision_ic95_high=pc[1],alertas=int(r.tp+r.fp)))
pd.DataFrame(w).to_csv(p/'testes/intervalos_confianca_metricas.csv',index=False)
pd.DataFrame(L).to_csv(p/'testes/testes_sustentacao.csv',index=False)
print(pd.DataFrame(L).to_string(index=False));print('intervalos',w)
