"""P2 · Features novas a partir das camadas (enriquecimento) e grupos de features.

Cada feature responde "eu sabia isso antes de carregar?": sim, vem do plano de camadas.
Convenção da EDA: nu_camada 1 = fundo do cestão; maior = topo.
"""
import numpy as np, pandas as pd, os
import config as C
from p1_dados import adiciona_historico

CHAVE = ['cd_corrida', 'nu_carregamento']


def features_camadas(d_dev_ids):
    """Cria features de posição/material. A lista de materiais 'frequentes' é escolhida só com o DEV."""
    cam = pd.read_csv(os.path.join(C.DADOS, 'infos_camadas.csv'), dtype={'cd_codigo_material': str})
    mat = pd.read_csv(os.path.join(C.DADOS, 'materiais_classificados.csv'), dtype={'material': str})
    cam = cam[cam.nu_carregamento.isin([1, 2])].copy()
    cam['nu_camada'] = cam.nu_camada.astype(int)
    cam = cam.merge(mat[['material', 'densidade', 'classe']], left_on='cd_codigo_material',
                    right_on='material', how='left')
    cam['peso_t'] = cam.qt_peso_carregamento / 1000
    cam['n'] = cam.groupby(CHAVE)['nu_camada'].transform('max')
    cam['pos'] = (cam.nu_camada - 1) / (cam.n - 1).replace(0, np.nan)   # 0 = fundo, 1 = topo
    cam['terco'] = pd.cut(cam.pos, [-0.01, 1/3, 2/3, 1.01], labels=['fundo', 'meio', 'topo']).astype(object)
    cam.loc[cam.pos.isna(), 'terco'] = 'meio'
    cam['vol'] = cam.peso_t / cam.densidade

    g = cam.groupby(CHAVE)
    out = pd.DataFrame(index=g.size().index)
    tot = g.peso_t.sum()
    # peso por terço (fração do cestão)
    for t in ['fundo', 'meio', 'topo']:
        out[f'pesofrac_{t}'] = cam[cam.terco == t].groupby(CHAVE).peso_t.sum().reindex(out.index).fillna(0) / tot
    # densidade média ponderada por volume em cada terço
    for t in ['fundo', 'meio', 'topo']:
        s = cam[cam.terco == t].groupby(CHAVE)
        out[f'dens_{t}'] = (s.peso_t.sum() / s.vol.sum()).reindex(out.index)
    # primeira e última camada
    fundo = cam[cam.nu_camada == 1].groupby(CHAVE).agg(peso_fundo=('peso_t', 'sum'), dens_camada1=('densidade', 'mean'))
    topo = cam[cam.nu_camada == cam.n].groupby(CHAVE).agg(peso_topo=('peso_t', 'sum'), dens_ultima=('densidade', 'mean'))
    out = out.join(fundo).join(topo)
    out['peso_max_camada'] = g.peso_t.max()
    out['peso_med_camada'] = g.peso_t.mean()
    out['frac_max_camada'] = out.peso_max_camada / tot
    # fração do peso total por material frequente (>=5% dos cestões do DEV)
    cam_dev = cam[cam.cd_corrida.isin(d_dev_ids)]
    freq = cam_dev.groupby('cd_codigo_material').apply(lambda s: s[CHAVE].drop_duplicates().shape[0])
    n_dev = cam_dev[CHAVE].drop_duplicates().shape[0]
    frequentes = freq[freq >= 0.05 * n_dev].index.tolist()
    pm = cam.pivot_table(index=CHAVE, columns='cd_codigo_material', values='peso_t', aggfunc='sum').fillna(0)
    for m in frequentes:
        out[f'mat_{m}'] = (pm[m] / tot).reindex(out.index).fillna(0)
    return out.reset_index(), [c for c in out.columns if c not in CHAVE]


def construir(d):
    """Retorna (df completo com todas as features, dict nome_do_grupo -> colunas)."""
    d = adiciona_historico(d)
    camf, cols_cam = features_camadas(set(d[d.dev].cd_corrida))
    d = d.merge(camf, on=CHAVE, how='left')
    grupos = {
        'tempo': C.TEMPO,
        'desfecho_c1': ['c1_carga_alta'],                                   # só cestão 2
        'feats_c1': ['c1_peso_total_t', 'c1_densidade', 'c1_regra_ok', 'c1_prop_leve'],  # só cestão 2
        'historico': ['h_prev_alta', 'h_taxa10', 'h_taxa50', 'h_desde_ult_alta'],
        'posicao': [c for c in cols_cam if c.startswith(('pesofrac_', 'dens_', 'peso_', 'frac_'))],
        'materiais': [c for c in cols_cam if c.startswith('mat_')],
    }
    d['regra_ok'] = d['regra_ok'].astype(float)
    return d, grupos
