"""EVCOMX | Random Forest de carga alta (PT-BR)

Instalação: pip install -r requirements.txt
Execução: streamlit run app.py
Variáveis de ambiente opcionais:
  EVCOMX_RF_DATA_DIR: pasta de previsões, exemplos, importâncias e config.json
  EVCOMX_RF_MODEL_DIR: pasta dos modelos random_forest_c1.joblib / c2.joblib

Os dois pipelines RF foram ajustados SOMENTE nos conjuntos de treino originais.
Os limiares foram escolhidos na validação, NÃO no teste futuro.
Os campos já calculados representam o estado do cestão no momento da previsão;
insumos que surjam após esse instante não podem ser usados para uma decisão real.
"""
import io
import json
import os
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

BASE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('EVCOMX_RF_DATA_DIR', BASE / 'assets'))
MODELS = Path(os.environ.get('EVCOMX_RF_MODEL_DIR', BASE / 'assets'))
ORANGE, GRAPHITE, PEACH, BG = '#F54000', '#222630', '#FDE8E2', '#FFFFFF'

st.set_page_config(page_title='EVCOMX | Random Forest', page_icon='🔥', layout='wide', initial_sidebar_state='expanded')
st.markdown("""<style>
html, body, .stApp {font-family: Arial, sans-serif;}
.stApp {background:#FFFFFF;color:#222630}
[data-testid="stSidebar"] {background:#222630}
[data-testid="stSidebar"] * {color:#FFFFFF !important}
.block-container {padding-top:1.6rem;max-width:1480px}
.banner {background:linear-gradient(115deg,#222630 0%,#222630 78%,#F54000 78%);
 padding:29px 30px;border-radius:14px;color:#FFFFFF;margin-bottom:22px}
.banner h1 {color:#FFFFFF;font-size:2rem;letter-spacing:-.035em;font-weight:800;margin:0}
.banner p {color:#FDE8E2;margin:7px 0 0;max-width:920px}
.infobox {background:#FFF5F0;border-left:5px solid #F54000;border-radius:7px;
 padding:13px 17px;margin:15px 0;}
[data-testid="stMetric"] {background:#FFFFFF;border:1px solid #EEE7E3;padding:14px;border-radius:11px}
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"]{background:#F54000;color:white;border-color:#F54000}
</style>""",unsafe_allow_html=True)


def br(x,dec=1):
    if x is None or not np.isfinite(float(x)):return '—'
    return f'{float(x):,.{dec}f}'.replace(',','X').replace('.',',').replace('X','.')


def pct(x):return f'{br(x*100,1)}%'


def money(x):return 'R$ '+br(x,0)


def hero(t,sub):st.markdown(f'<div class="banner"><h1>{escape(t)}</h1><p>{escape(sub)}</p></div>',unsafe_allow_html=True)


def note(t):st.markdown(f'<div class="infobox">{escape(t)}</div>',unsafe_allow_html=True)


def styled(fig,height=365):
    fig.update_layout(template='plotly_white',font=dict(family='Arial',color=GRAPHITE),
       paper_bgcolor=BG,plot_bgcolor=BG,height=height,margin=dict(l=10,r=10,t=50,b=42),
       legend=dict(orientation='h',y=-0.2))
    return fig


@st.cache_data(show_spinner=False)
def get_data(directory):
    p=Path(directory)
    files={'test':'previsoes_teste_rf.csv','valid':'previsoes_validacao_rf.csv',
           'examples':'exemplos_teste_rf.csv','importance':'importancias_rf.csv'}
    ds={k:pd.read_csv(p/f) for k,f in files.items()}
    for k in ['test','valid','examples']:ds[k]['t']=pd.to_datetime(ds[k]['t'],errors='coerce')
    for k in ['test','valid']:
        ds[k]['carga_alta']=ds[k]['carga_alta'].astype(int)
        ds[k]['parada_s']=pd.to_numeric(ds[k]['parada_s'],errors='raise').fillna(0)
        ds[k]['p']=pd.to_numeric(ds[k]['p'],errors='raise')
        ds[k]['alerta_original']=ds[k]['alerta_original'].astype(int)
    return ds


@st.cache_resource(show_spinner=False)
def get_models(directory,config_path):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    models={}
    for c in (1,2):
        model=joblib.load(Path(directory)/f'random_forest_c{c}.joblib')
        cols=cfg['cestoes'][str(c)]['features']
        if list(model.feature_names_in_)!=cols:
            raise ValueError(f'Ordem de variáveis do Random Forest do cestão {c} não corresponde ao config.json')
        if model.named_steps['model'].n_estimators!=180:
            raise ValueError(f'Modelo de cestão {c} não tem 180 árvores.')
        models[c]=model
    return cfg,models


def subset(data,c,kind='test'):
    return data[kind].loc[data[kind]['cestao'].eq(c)].copy().sort_values('t')


def predict_rf(frame,c,cfg,models):
    cols=cfg['cestoes'][str(c)]['features']
    missing=[col for col in cols if col not in frame.columns]
    if missing:raise ValueError('Faltam colunas obrigatórias: '+', '.join(missing))
    xx=frame.loc[:,cols].copy()
    for col in cols:
        x=xx[col]
        if not pd.api.types.is_numeric_dtype(x):
            x=x.astype('string').str.replace(',','.',regex=False)
        xx[col]=pd.to_numeric(x,errors='raise')
    xx=xx.replace([np.inf,-np.inf],np.nan)
    if c==2:
        a=xx['carga_alta_c1_ctx']
        if a.isna().any() or not a.isin([0,1]).all():
            raise ValueError('Para o cestão 2, carga_alta_c1_ctx precisa ser o resultado observado 0 ou 1 do cestão 1.')
    return models[c].predict_proba(xx)[:,1]


def metrics(df,threshold=None,original=False):
    y=df['carga_alta'].to_numpy(dtype=int);p=df['p'].to_numpy(dtype=float)
    alert=df['alerta_original'].to_numpy(dtype=int) if original else (p>=float(threshold)).astype(int)
    tp=int(((alert==1)&(y==1)).sum());fp=int(((alert==1)&(y==0)).sum())
    fn=int(((alert==0)&(y==1)).sum());tn=int(((alert==0)&(y==0)).sum())
    cap=float(df.loc[(alert==1)&(df.carga_alta==1),'parada_s'].sum())/60
    return {'n':len(df),'eventos':int(sum(y)),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'alertas':tp+fp,
        'recall':tp/(tp+fn) if tp+fn else 0.,'precisao':tp/(tp+fp) if tp+fp else 0.,
        'roc':roc_auc_score(y,p) if len(set(y))>1 else float('nan'),
        'pr':average_precision_score(y,p) if len(set(y))>1 else float('nan'),
        'capturados_min':cap,'paradas_total_min':float(df.loc[df.carga_alta.eq(1),'parada_s'].sum()/60)}


def economic(s,eff,fp_cost,money_min,df):
    gross=eff*s['capturados_min'];cost=fp_cost*s['fp'];net=gross-cost
    dts=pd.to_datetime(df['t'].dropna())
    if len(dts)>=2:
        n_months=max((dts.max()-dts.min()).total_seconds()/86400/30.4375,1/30.4375)
        yearly=net/n_months*12*money_min
    else:yearly=float('nan')
    return {'bruto':gross,'custo':cost,'liquido':net,'anual_reais':yearly,
            'mensal_reais':yearly/12,'equilibrio':cost/s['capturados_min'] if s['capturados_min']>0 else np.inf}


def select_c(key):
    return st.selectbox('Carregamento', [1,2],format_func=lambda c:f'Cestão {c}',key=key)


def view_overview(data,cfg):
    hero('Random Forest · visão geral','Risco de carga alta nos carregamentos 1 e 2, com avaliação temporal fora da amostra.')
    tests=data['test'];a,b,c,d=st.columns(4)
    a.metric('Corridas no teste',str(tests.cd_corrida.nunique()))
    b.metric('Cestões avaliados',str(len(tests)))
    c.metric('Eventos de carga alta',str(int(tests.carga_alta.sum())))
    d.metric('Árvores por modelo','180')
    note('Acertar a maioria dos casos não prova que o modelo detecta eventos raros. Por isso priorizamos recall, precisão e PR-AUC, além de ROC-AUC.')
    st.subheader('Resultados do teste futuro · limiares definidos na validação')
    rows=[]
    for cestao in [1,2]:
        s=metrics(subset(data,cestao),original=True)
        rows.append({'Cestão':cestao,'Observações':s['n'],'Eventos':s['eventos'],
          'VP':s['tp'],'FP':s['fp'],'FN':s['fn'],'VN':s['tn'],
          'Recall':pct(s['recall']),'Precisão':pct(s['precisao']),
          'ROC-AUC':br(s['roc'],3),'PR-AUC':br(s['pr'],3),'Min. sinalizados':br(s['capturados_min'])})
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    st.markdown('**Como foi treinado:** o Random Forest utiliza 180 árvores, mínimo de 12 amostras por folha, 65% de variáveis candidatas em cada divisão e imputação por mediana aprendida apenas no treino. Um modelo independente por cestão, com **56 e 59 variáveis**, respectivamente.')
    c1,c2=st.columns(2)
    for c,container in [(1,c1),(2,c2)]:
        info=cfg['cestoes'][str(c)];df=subset(data,c)
        with container:
            st.markdown(f'**Cestão {c} · amostragem**')
            st.write(f"Treino: {info['treino_n']} | Validação: {info['validacao_n']} | Teste: {info['teste_n']}")
            st.caption(f"Período do teste: {df.t.min():%d/%m/%Y} a {df.t.max():%d/%m/%Y}. Limiar de validação: {br(info['limiar'],3)}")
    st.info('O conjunto de teste não participou do ajuste dos modelos nem da escolha dos limiares. Os números no painel são previsões historicamente fora da amostra.')


PHYSICAL={'peso_total_t','volume_estimado_m3','densidade_media','n_camadas','prop_leve','prop_pesado','prop_misto',
          'hhi_materiais','energia_por_ton','rendimento_medio','carga_alta_c1_ctx','taxa_ca_ult10'}

def nice_name(name):
    labels={'peso_total_t':'Peso total do cestão (t)','volume_estimado_m3':'Volume estimado (m³)',
      'densidade_media':'Densidade média','n_camadas':'Número de camadas','prop_leve':'Fração de material leve',
      'prop_pesado':'Fração de material pesado','prop_misto':'Fração de material misto',
      'hhi_materiais':'Concentração dos materiais (HHI)','energia_por_ton':'Energia associada (kWh/t)',
      'rendimento_medio':'Rendimento metálico médio','taxa_ca_ult10':'Carga alta nas últimas 10 corridas',
      'carga_alta_c1_ctx':'O primeiro cestão teve carga alta? (0 ou 1)'}
    return labels.get(name,name.replace('_',' ').capitalize())+f'  [{name}]'


def view_simulator(data,cfg,models):
    hero('Simulador de alerta','Use um exemplo real da amostra ou informe as variáveis do carregamento para estimar o risco.')
    c=select_c('sim_c')
    features=cfg['cestoes'][str(c)]['features']
    examples=data['examples'].query('cestao==@c').copy()
    sample_id=st.selectbox('Escolha uma corrida histórica como ponto de partida',examples.cd_corrida.astype(str).tolist(),key=f'ex_sample_{c}')
    sample=examples[examples.cd_corrida.astype(str)==sample_id].iloc[0]
    note('O formulário começa com valores reais de um carregamento do teste. Alterar apenas uma variável pode criar um cenário fisicamente incoerente. A saída é um escore estatístico, não uma ordem automática para alterar a operação.')
    if c==2:
        confirmation=st.checkbox('Confirmo que o resultado do primeiro cestão já é conhecido antes desta previsão.',value=False)
        st.caption('O Random Forest do segundo cestão utiliza carga_alta_c1_ctx. Não é permitido usar esse resultado antes que o primeiro cestão termine.')
    else:confirmation=True
    default=float(cfg['cestoes'][str(c)]['limiar'])
    with st.form(f'rf_form_{c}_{sample_id}'):
        st.markdown('**Variáveis principais**')
        values={};primary=[x for x in features if x in PHYSICAL]
        advanced=[x for x in features if x not in PHYSICAL]
        cols=st.columns(2)
        for i,key in enumerate(primary):
            v=sample[key];v=float(v) if pd.notna(v) else 0.0
            with cols[i%2]:
                values[key]=st.number_input(nice_name(key),value=v,format='%.4f',key=f'sim_{c}_{sample_id}_{key}')
        with st.expander(f'Mais {len(advanced)} variáveis do modelo (avançado)',expanded=False):
            st.caption('Variáveis derivadas de composição, posição, histórico e horário. Valores ausentes no exemplo são imputados por mediana do treino.')
            subcols=st.columns(2)
            for i,key in enumerate(advanced):
                v=sample[key]
                with subcols[i%2]:
                    if pd.isna(v):
                        missing=st.checkbox('Usar ausente · '+key,value=True,key=f'null_{c}_{sample_id}_{key}')
                        values[key]=np.nan if missing else st.number_input(nice_name(key),value=0.0,format='%.4f',key=f'sim_{c}_{sample_id}_{key}')
                    else:
                        values[key]=st.number_input(nice_name(key),value=float(v),format='%.4f',key=f'sim_{c}_{sample_id}_{key}')
        threshold=st.slider('Limiar de alerta · escore bruto do Random Forest',0.,1.,default,.005)
        submitted=st.form_submit_button('Prever carga alta',type='primary',use_container_width=True)
    if not submitted:return
    if not confirmation:
        st.error('Confirme que o resultado do primeiro cestão já foi observado para utilizar o modelo do segundo cestão.');return
    if c==2 and values['carga_alta_c1_ctx'] not in (0.,1.):
        st.error('O resultado confirmado do cestão 1 precisa ser 0 ou 1.');return
    if 'prop_leve' in values and 'prop_pesado' in values and 'prop_misto' in values:
        if abs(sum(values[k] for k in ['prop_leve','prop_pesado','prop_misto'])-1)>.06:
            st.warning('As proporções dos materiais não somam aproximadamente 1. Verifique o cenário.')
    try:risk=float(predict_rf(pd.DataFrame([values]),c,cfg,models)[0])
    except Exception as exc:st.error(f'Falha na previsão: {exc}');return
    a,b,d=st.columns(3)
    a.metric('Risco estimado',pct(risk));b.metric('Limiar atual',pct(threshold));d.metric('Decisão','ALERTA' if risk>=threshold else 'SEM ALERTA')
    st.progress(min(100,max(0,int(round(100*risk)))))
    if risk>=threshold:st.error('ALERTA: carregamento acima do limiar de risco selecionado.')
    else:st.success('Risco abaixo do limiar selecionado.')
    st.caption('Não há calibrador de probabilidade validado para este Random Forest. O valor apresentado é a saída bruta do classificador; não deve ser interpretado como frequência empírica garantida.')
    original=subset(data,c).set_index('cd_corrida')
    if sample_id in original.index:
        matched=original.loc[sample_id]
        st.caption(f"Consulta ao resultado histórico: carga alta observada = {int(matched['carga_alta'])}; escore original = {pct(float(matched['p']))}. A consulta ao desfecho é apenas para explicar o exemplo e não entra na previsão do cestão 1.")


SAFE_METADATA={'cd_corrida','cestao','t','c1_resultado_confirmado'}
FORBIDDEN_TOKENS=('parada','desfecho','carga_alta','resultado','observado','probabilidade','risco_previsto','target','label','predicao','previsao')

def validate_batch_columns(df,features,c):
    if not df.columns.is_unique:raise ValueError('Há nomes de colunas duplicados.')
    missing=[f for f in features if f not in df]
    if missing:raise ValueError('Faltam variáveis obrigatórias: '+', '.join(missing))
    extras=[f for f in df if f not in set(features)|SAFE_METADATA]
    if extras:raise ValueError('Colunas extras/proibidas: '+', '.join(extras))
    if c==2:
        if 'c1_resultado_confirmado' not in df:raise ValueError('Inclua c1_resultado_confirmado=sim para cada linha do cestão 2.')
        valid=df['c1_resultado_confirmado'].astype('string').str.strip().str.lower().isin(['sim','true','1','yes'])
        if not valid.all():raise ValueError('O desfecho do 1º cestão precisa estar confirmado para todas as linhas.')
    if 'cestao' in df and not pd.to_numeric(df['cestao'],errors='coerce').eq(c).all():
        raise ValueError('O arquivo contém o número de cestão diferente do selecionado.')


def view_batch(data,cfg,models):
    hero('Previsão em lote','Carregue um arquivo CSV com as variáveis anteriores à decisão e exporte riscos e alertas.')
    c=select_c('batch_c')
    cols=cfg['cestoes'][str(c)]['features'];th=st.slider('Limiar do risco bruto',0.,1.,float(cfg['cestoes'][str(c)]['limiar']),.005)
    template=pd.DataFrame([{name:np.nan for name in cols}]);template.insert(0,'cestao',c)
    if c==2:template['c1_resultado_confirmado']='sim'
    st.download_button('Baixar modelo de entrada CSV',template.to_csv(index=False,sep=';',decimal=',').encode('utf-8-sig'),file_name=f'entrada_RF_c{c}.csv',mime='text/csv')
    file=st.file_uploader('CSV de carregamentos (máximo 20 MB)',type='csv',key='batch_csv')
    if file is None:return
    try:
        if len(file.getvalue())>20_000_000:raise ValueError('Arquivo grande demais para a demonstração.')
        df=pd.read_csv(io.BytesIO(file.getvalue()),sep=None,engine='python',encoding='utf-8-sig')
        df.columns=[x.strip() for x in df.columns]
        if not 0<len(df)<=100_000:raise ValueError('Envie entre 1 e 100.000 registros.')
        validate_batch_columns(df,cols,c)
        p=predict_rf(df,c,cfg,models)
        out=df[[x for x in ['cd_corrida','cestao','t'] if x in df]].copy()
        out['risco_rf']=np.round(p,6);out['alerta']=(p>=th).astype(int)
    except Exception as e:st.error(str(e));return
    a,b,d=st.columns(3);a.metric('Registros',len(df));b.metric('Alertas',int(out.alerta.sum()));d.metric('Taxa de alertas',pct(out.alerta.mean()))
    st.dataframe(out,hide_index=True,use_container_width=True)
    st.download_button('Baixar resultados CSV',out.to_csv(index=False,sep=';',decimal=',').encode('utf-8-sig'),file_name=f'previsoes_RF_c{c}.csv',mime='text/csv',type='primary')
    st.caption('Os CSVs aceitam valores ausentes dentro das colunas obrigatórias, tratados pelo imputador de mediana do treino. Colunas de resultado/parada não entram no modelo. O segundo cestão requer o evento confirmado do primeiro.')


def threshold_curve(df,eff,fp_cost):
    arr=[]
    for th in np.linspace(0.,1.,101):
        s=metrics(df,th)
        arr.append({'Limiar':th,'Ganho líquido (min)':s['capturados_min']*eff-s['fp']*fp_cost})
    return pd.DataFrame(arr)


def view_business(data,cfg):
    hero('Business case interativo','Simulação de economia potencial usando as previsões históricas do Random Forest.')
    st.markdown('**Ganho líquido (min) = eficácia × minutos de parada dos verdadeiros positivos − custo_fp × número de falsos positivos.** Falsos negativos não adicionam ganhos ou custos.')
    a,b,d=st.columns(3)
    eff=a.slider('Eficácia da ação (%)',0,100,100,5)/100
    fp_cost=b.slider('Custo por falso alarme (min)',0.,5.,1.,.25)
    val_min=d.number_input('Valor de produção por minuto (R$)',min_value=0.,value=2500.,step=100.)
    st.caption('Parâmetros hipotéticos: eficácia de 100% e R$ 2.500/min não são resultados medidos. É necessário validar esses valores com a usina.')
    thresholds={};rows=[];parts=[];econ={}
    controls=st.columns(2)
    for c in (1,2):
        default=float(cfg['cestoes'][str(c)]['limiar'])
        with controls[c-1]:
            thresholds[c]=st.slider(f'Cestão {c} · limiar de alerta',0.,1.,default,.005,key=f'econ_th_{c}')
        test=subset(data,c);s=metrics(test,thresholds[c]);e=economic(s,eff,fp_cost,val_min,test);econ[c]=(s,e,test)
        rows.append({'Cestão':c,'Alertas':s['alertas'],'VP':s['tp'],'FP':s['fp'],'Paradas capturadas (min)':s['capturados_min'],
          'Minutos por alarmes':e['custo'],'Ganho líquido (min)':e['liquido'],'Cenário R$/mês':e['mensal_reais'],
          'Cenário R$/ano':e['anual_reais']})
        for k,v in [('Capturados com eficácia',e['bruto']),('Custo de FP',-e['custo']),('Líquido',e['liquido'])]:parts.append({'Cestão':f'Cestão {c}','Componente':k,'Minutos':v})
    st.dataframe(pd.DataFrame(rows).assign(**{'Paradas capturadas (min)':lambda t:t['Paradas capturadas (min)'].map(br),
        'Minutos por alarmes':lambda t:t['Minutos por alarmes'].map(br),'Ganho líquido (min)':lambda t:t['Ganho líquido (min)'].map(br),
        'Cenário R$/mês':lambda t:t['Cenário R$/mês'].map(money),'Cenário R$/ano':lambda t:t['Cenário R$/ano'].map(money)}),hide_index=True,use_container_width=True)
    st.plotly_chart(styled(px.bar(pd.DataFrame(parts),x='Cestão',y='Minutos',color='Componente',barmode='group',
        color_discrete_map={'Capturados com eficácia':ORANGE,'Custo de FP':GRAPHITE,'Líquido':'#F7AA8C'},
        title='Paradas sinalizadas, custo e ganho líquido hipotético')),use_container_width=True)
    c=select_c('business_chart_c')
    v=threshold_curve(subset(data,c,'valid'),eff,fp_cost).assign(Amostra='Validação (corte selecionado)')
    t=threshold_curve(subset(data,c),eff,fp_cost).assign(Amostra='Teste futuro (sensibilidade)')
    fig=px.line(pd.concat([v,t]),x='Limiar',y='Ganho líquido (min)',color='Amostra',color_discrete_sequence=[ORANGE,GRAPHITE],title=f'Ganho por limiar · cestão {c}')
    fig.add_vline(x=cfg['cestoes'][str(c)]['limiar'],line_dash='dot',line_color='#B74D31')
    st.plotly_chart(styled(fig),use_container_width=True)
    s,e,_=econ[c]
    st.subheader(f'Sensibilidade à eficácia · cestão {c}')
    sens=[]
    for ef in [1.,.7,.5,.3]:
        ee=economic(s,ef,fp_cost,val_min,subset(data,c))
        sens.append({'Eficácia':pct(ef),'Minutos líquidos':br(ee['liquido']),'R$/mês (cenário)':money(ee['mensal_reais'])})
    st.dataframe(pd.DataFrame(sens),hide_index=True,use_container_width=True)
    st.metric('Eficácia mínima para empatar',pct(e['equilibrio']) if np.isfinite(e['equilibrio']) else 'Sem captura')
    note('Alterar um limiar usando o teste é somente análise exploratória; não reutilize o teste para escolher uma nova política e depois declarar validação independente. Anualização linear com base no intervalo efetivo do teste, sem efeitos de sazonalidade ou escala.')


def view_performance(data,cfg):
    hero('Desempenho e explicabilidade','Análise do conjunto de teste, distribuição do risco e importância preditiva das variáveis.')
    c=select_c('perf_c');d=subset(data,c)
    original=st.checkbox('Usar o limiar escolhido na validação',value=True)
    threshold=float(cfg['cestoes'][str(c)]['limiar']) if original else st.slider('Novo limiar (exploratório)',0.,1.,float(cfg['cestoes'][str(c)]['limiar']),.005)
    s=metrics(d,original=original,threshold=threshold)
    cols=st.columns(4)
    for block,key,val in zip(cols,['Recall','Precisão','ROC-AUC','PR-AUC'],[pct(s['recall']),pct(s['precisao']),br(s['roc'],3),br(s['pr'],3)]):block.metric(key,val)
    x,y=st.columns(2)
    with x:
        fig=go.Figure(go.Heatmap(z=[[s['tn'],s['fp']],[s['fn'],s['tp']]],
            x=['Sem alerta','Com alerta'],y=['Sem carga alta','Com carga alta'],
            text=[[s['tn'],s['fp']],[s['fn'],s['tp']]],texttemplate='%{text}',textfont=dict(size=20),
            colorscale=[[0,'#FFF6F2'],[1,ORANGE]],showscale=False))
        fig.update_layout(title='Matriz de confusão',yaxis_autorange='reversed')
        st.plotly_chart(styled(fig),use_container_width=True)
    with y:
        pr,re,_=precision_recall_curve(d.carga_alta,d.p)
        fig=go.Figure();fig.add_trace(go.Scatter(x=re,y=pr,mode='lines',line=dict(color=ORANGE,width=3),name='Random Forest'))
        fig.add_hline(y=float(d.carga_alta.mean()),line_dash='dot',line_color=GRAPHITE)
        fig.update_layout(title='Precisão × Recall',xaxis_title='Recall',yaxis_title='Precisão')
        st.plotly_chart(styled(fig),use_container_width=True)
    a,b=st.columns(2)
    with a:
        temp=d.copy();temp['faixa']=pd.qcut(temp.p,q=min(8,max(3,len(temp)//30)),duplicates='drop')
        grouped=temp.groupby('faixa',observed=True).agg(risco=('p','mean'),evento=('carga_alta','mean')).reset_index()
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(color='#969696',dash='dot'),name='Ideal'))
        fig.add_trace(go.Scatter(x=grouped.risco,y=grouped.evento,mode='lines+markers',line=dict(color=ORANGE),name='RF bruto'))
        fig.update_layout(title='Calibração descritiva (teste)',xaxis_title='Escore RF médio',yaxis_title='Frequência de evento observada',xaxis_range=[0,1],yaxis_range=[0,1])
        st.plotly_chart(styled(fig),use_container_width=True)
    with b:
        imp=data['importance'].query('cestao==@c').nlargest(12,'importancia_media').sort_values('importancia_media')
        st.plotly_chart(styled(px.bar(imp,x='importancia_media',y='feature',orientation='h',color_discrete_sequence=[ORANGE],
            title='Importância por permutação · validação')),use_container_width=True)
    note('Importância não indica causalidade. A importância foi estimada na validação por permutação das variáveis, não mede o efeito de uma intervenção. As probabilidades do Random Forest não passaram por calibração adicional validada.')


def view_limits(data,cfg):
    hero('Limitações e próximos passos','Como interpretar os resultados antes de considerar uma aplicação operacional.')
    df=data['test'];a,b,c=st.columns(3)
    a.metric('Corridas no teste',df.cd_corrida.nunique());b.metric('Carregamentos no teste',len(df));c.metric('Eventos no teste',int(df.carga_alta.sum()))
    st.markdown('''**Limites conhecidos:**
- Os modelos foram treinados exclusivamente com dados de treino; os limiares foram escolhidos na validação. O teste foi preservado.
- As previsões são condicionadas à disponibilidade de **todas as informações antes do instante da decisão**. Tempo de montagem, sequência completa e informação do cestão 1 podem não estar disponíveis em decisões antecipadas.
- O cestão 2 usa `carga_alta_c1_ctx`; a variável só é admissível depois de confirmado o resultado do primeiro cestão.
- Histórico recente, classificação de materiais e composição precisam de dados atualizados para novos carregamentos. Ausentes são imputados pela mediana do treino, conforme o pipeline original.
- O escore bruto do Random Forest não é probabilidade calibrada. É preciso acompanhar confiabilidade, mudança de processo e precisão dos alertas ao longo do tempo.
- Ganho de minutos e receita anualizados são **cenários contrafactuais**, não economia real. Eficácia da ação, custo do falso positivo e valor do minuto devem ser confirmados com a usina.
- Modificar variáveis derivadas isoladamente pode produzir carregamentos impossíveis; só intervenções fisicamente seguras e aprovadas devem ser consideradas.''')
    st.markdown('**Próximos passos:** validar com os engenheiros o instante exato de previsão, observar o desempenho em dados novos sem reajustar o modelo, calibrar os escores se necessário e avaliar um piloto supervisionado com medição do tempo realmente evitado.')
    st.warning('Ferramenta demonstrativa para apoio analítico. Não executar comandos industriais nem alterar receitas e procedimentos de segurança automaticamente.')


def main():
    st.sidebar.markdown('<div style="font-weight:900;font-size:29px;padding:10px 0;">EV<span style="color:#F54000">COMX</span></div>',unsafe_allow_html=True)
    st.sidebar.caption('FUNDO DO AMANHÃ · RANDOM FOREST')
    pages=['Visão geral','Simulador de alerta','Previsão em lote','Business case interativo','Desempenho e explicabilidade','Limitações e próximos passos']
    page=st.sidebar.radio('Navegação',pages)
    st.sidebar.divider()
    st.sidebar.markdown('**Modelo selecionado**')
    st.sidebar.caption('Random Forest · 180 árvores · um modelo para cada cestão')
    st.sidebar.caption('Treino e validação cronológicos | Teste preservado')
    try:
        data=get_data(str(DATA));cfg,models=get_models(str(MODELS),str(DATA/'config.json'))
    except Exception as exc:
        st.error('Não foi possível carregar a modelagem Random Forest. Confirme os caminhos configurados em EVCOMX_RF_DATA_DIR e EVCOMX_RF_MODEL_DIR.')
        st.code(str(exc));st.stop()
    if page==pages[0]:view_overview(data,cfg)
    elif page==pages[1]:view_simulator(data,cfg,models)
    elif page==pages[2]:view_batch(data,cfg,models)
    elif page==pages[3]:view_business(data,cfg)
    elif page==pages[4]:view_performance(data,cfg)
    else:view_limits(data,cfg)
    st.divider()
    st.caption('EVCOMX · Protótipo analítico · Random Forest · Não representa economia comprovada e não substitui a operação segura.')

if __name__=='__main__':main()
