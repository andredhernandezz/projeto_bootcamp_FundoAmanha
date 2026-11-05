"""
01_features.py  -  Feature Engineering (guia EVCOMX FE/FS) por CESTAO (carregamento 1 e 2).

Entrada : pasta com os CSVs originais (argumento 1)
Saida   : outputs/dados/features_cestao.csv   (1 linha por corrida x carregamento)

Regras anti-vazamento (ver dicionario_colunas_modelagem.csv):
  - parada_s / qt_segundos_* sao POS-EVENTO -> nunca viram feature.
  - is_carga_alta (da corrida) inclui o outro cestao -> nunca feature.
  - gusa liquido (carregamento 3) entra DEPOIS dos cestoes -> marcado como grupo 'gusa' (cenario B).
  - historico usa shift(1): so corridas anteriores.
  - carga_alta do cestao 1 pode ser feature do cestao 2 (ja aconteceu quando o 2 e despejado).
"""
import sys, os
import numpy as np
import pandas as pd
import holidays

SRC = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ carga
camadas = pd.read_csv(f"{SRC}/infos_camadas.csv").drop_duplicates()
camadas["dt"] = pd.to_datetime(camadas["dt_hora_consumo"])
camadas["peso_t"] = camadas["qt_peso_carregamento"] / 1000.0
camadas["nu_carregamento"] = camadas["nu_carregamento"].astype(int)

mat = pd.read_csv(f"{SRC}/infos_materiais_limpo.csv")
mat.columns = ["cod", "energia_kwh_t", "dens_t_m3", "rend"]
mat["cod"] = mat["cod"].astype(str)
for c in ["energia_kwh_t", "dens_t_m3", "rend"]:
    mat[c] = pd.to_numeric(mat[c], errors="coerce")        # insumos (COQUE/CAL) viram NaN
mat = mat.dropna(subset=["dens_t_m3"]).set_index("cod")

cls = pd.read_csv(f"{SRC}/dicionario_tipo_material.csv")
cls["cd_codigo_material"] = cls["cd_codigo_material"].astype(str)
cls = cls.set_index("cd_codigo_material")["tp_material"].str.lower()

camadas["cod"] = camadas["cd_codigo_material"].astype(str)
camadas["classe"] = camadas["cod"].map(cls)
camadas["dens"] = camadas["cod"].map(mat["dens_t_m3"])
camadas["energia"] = camadas["cod"].map(mat["energia_kwh_t"])
camadas["rend"] = camadas["cod"].map(mat["rend"])
camadas["vol_m3"] = camadas["peso_t"] / camadas["dens"]

resp = pd.read_csv(f"{SRC}/VariavelResposta.csv")

# ------------------------------------------------------------------ materiais frequentes (guia 4.3)
solid = camadas[camadas["nu_carregamento"].isin([1, 2])].copy()
n_cest = solid.groupby(["cd_corrida", "nu_carregamento"]).ngroups
freq = solid.groupby("cod").apply(lambda d: d[["cd_corrida", "nu_carregamento"]].drop_duplicates().shape[0]) / n_cest
FREQ_OK = freq[freq > 0.10].index.tolist()               # > 10% dos cestoes ; o resto vira prop_OUTROS


def feats_cestao(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("nu_camada")
    p = g["peso_t"].values
    peso_tot = p.sum()
    pos_mask = p > 0
    gp = g[pos_mask]                                        # camadas realmente carregadas
    n = len(gp)
    out = {}
    out["peso_total_t"] = peso_tot
    out["n_camadas"] = len(g)
    out["camadas_peso_zero"] = int((~pos_mask).sum())
    out["n_materiais"] = gp["cod"].nunique()
    out["n_baias"] = gp["cd_baia"].nunique()

    # --- massa / volume
    com_d = gp[gp["dens"].notna()]
    vol = com_d["vol_m3"].sum()
    out["volume_estimado_m3"] = vol
    out["peso_sem_densidade_t"] = gp.loc[gp["dens"].isna(), "peso_t"].sum()
    out["densidade_media"] = com_d["peso_t"].sum() / vol if vol > 0 else np.nan
    for c in ["leve", "misto", "pesado"]:
        out[f"peso_{c}_t"] = gp.loc[gp["classe"] == c, "peso_t"].sum()
        out[f"prop_{c}"] = out[f"peso_{c}_t"] / peso_tot if peso_tot > 0 else np.nan
    out["vol_leve_m3"] = com_d.loc[com_d["classe"] == "leve", "vol_m3"].sum()
    out["prop_vol_leve"] = out["vol_leve_m3"] / vol if vol > 0 else np.nan

    # --- rendimento / energia (guia 3.3)
    com_r = gp[gp["rend"].notna()]
    out["rendimento_medio"] = (com_r["peso_t"] * com_r["rend"]).sum() / com_r["peso_t"].sum() if len(com_r) else np.nan
    com_e = gp[gp["energia"].notna()]
    out["energia_estimada_mwh"] = (com_e["peso_t"] * com_e["energia"]).sum() / 1000.0
    out["energia_por_ton"] = (com_e["peso_t"] * com_e["energia"]).sum() / com_e["peso_t"].sum() if len(com_e) else np.nan

    # --- composicao / diversidade (guia 4)
    prop_mat = gp.groupby("cod")["peso_t"].sum() / peso_tot
    out["hhi_materiais"] = float((prop_mat ** 2).sum())
    for m in FREQ_OK:
        out[f"prop_{m}"] = float(prop_mat.get(m, 0.0))
    out["prop_OUTROS"] = float(prop_mat[~prop_mat.index.isin(FREQ_OK)].sum())

    # --- posicao / ordem (guia 5)
    if n >= 2:
        pos = np.linspace(0, 1, n)
    else:
        pos = np.array([0.5])
    cl = gp["classe"].values
    pw = gp["peso_t"].values
    out["base_leve"] = int(cl[0] == "leve") if n else 0
    out["topo_leve"] = int(cl[-1] == "leve") if n else 0
    out["base_pesado"] = int(cl[0] == "pesado") if n else 0
    out["topo_pesado"] = int(cl[-1] == "pesado") if n else 0
    out["regra_ok"] = int(out["base_leve"] and out["topo_leve"])
    out["violacao_regra"] = int(out["base_pesado"] or out["topo_pesado"])
    topo = pos >= 0.7
    base = pos <= 0.3
    meio = (pos > 0.3) & (pos < 0.7)
    out["peso_topo_30pct_t"] = pw[topo].sum()
    out["peso_base_30pct_t"] = pw[base].sum()
    out["prop_leve_topo"] = pw[topo & (cl == "leve")].sum() / pw[topo].sum() if pw[topo].sum() > 0 else 0.0
    out["prop_pesado_base"] = pw[base & (cl == "pesado")].sum() / pw[base].sum() if pw[base].sum() > 0 else 0.0
    pes = cl == "pesado"
    out["pos_media_pesado"] = (pos[pes] * pw[pes]).sum() / pw[pes].sum() if pw[pes].sum() > 0 else 0.5
    lev = cl == "leve"
    out["leve_no_meio_pct"] = pw[meio & lev].sum() / pw[lev].sum() if pw[lev].sum() > 0 else 0.0
    out["pesado_fora_meio_pct"] = pw[~meio & pes].sum() / pw[pes].sum() if pw[pes].sum() > 0 else 0.0

    # --- sequencia temporal (guia 5.3)
    ts = g["dt"].dropna().sort_values()
    out["inicio_cestao"] = ts.iloc[0] if len(ts) else pd.NaT
    out["tempo_montagem_min"] = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 60 if len(ts) > 1 else np.nan
    out["n_timestamps"] = len(ts)
    out["intervalo_medio_min"] = out["tempo_montagem_min"] / (len(ts) - 1) if len(ts) > 2 else np.nan
    out["_mats"] = frozenset(gp["cod"].unique())
    return pd.Series(out)


print("construindo features por cestao...")
F = solid.groupby(["cd_corrida", "nu_carregamento"]).apply(feats_cestao, include_groups=False).reset_index()

# ------------------------------------------------------------------ carregamento 3 (gusa)  -> grupo 'gusa'
c3 = camadas[camadas["nu_carregamento"] == 3].groupby("cd_corrida")["peso_t"].sum().rename("peso_carreg3_t")
F = F.merge(c3, on="cd_corrida", how="left")
F["peso_carreg3_t"] = F["peso_carreg3_t"].fillna(0.0)
F["tem_gusa_liquido"] = (F["peso_carreg3_t"] > 0).astype(int)

# ------------------------------------------------------------------ horario: cestao sem timestamp herda o da corrida
inicio_corr = camadas.groupby("cd_corrida")["dt"].min().rename("inicio_corrida")
F = F.merge(inicio_corr, on="cd_corrida", how="left")
F["horario_herdado"] = F["inicio_cestao"].isna().astype(int)
F["inicio_cestao"] = F["inicio_cestao"].fillna(F["inicio_corrida"])
F["inicio_cestao"] = F["inicio_cestao"].dt.tz_localize(None) if F["inicio_cestao"].dt.tz is not None else F["inicio_cestao"]
F["inicio_corrida"] = F["inicio_corrida"].dt.tz_localize(None) if F["inicio_corrida"].dt.tz is not None else F["inicio_corrida"]

# ------------------------------------------------------------------ alvo por cestao (+ duracao da parada, so p/ negocio)
F = F.merge(resp, on="cd_corrida", how="left")
F["y"] = np.where(F["nu_carregamento"] == 1, F["is_carga_alta_carregamento_1"], F["is_carga_alta_carregamento_2"]).astype(int)
F["parada_s_cestao"] = np.where(F["nu_carregamento"] == 1, F["qt_segundos_duracao_parada_carregamento_1"],
                                F["qt_segundos_duracao_parada_carregamento_2"]).astype(float)
F = F.drop(columns=["is_carga_alta_carregamento_1", "is_carga_alta_carregamento_2",
                    "qt_segundos_duracao_parada_carregamento_1", "qt_segundos_duracao_parada_carregamento_2",
                    "qt_segundos_parada_carga_alta"])
F = F.rename(columns={"is_carga_alta": "y_corrida"})

# ------------------------------------------------------------------ temporais / feriados
t = F["inicio_cestao"]
F["hora"] = t.dt.hour
F["dia_semana"] = t.dt.dayofweek
F["fim_semana"] = (F["dia_semana"] >= 5).astype(int)
F["turno"] = pd.cut(F["hora"], [-1, 5, 13, 21, 24], labels=["C", "A", "B", "C2"]).astype(str).replace({"C2": "C"})
F["hora_sin"] = np.sin(2 * np.pi * F["hora"] / 24)
F["hora_cos"] = np.cos(2 * np.pi * F["hora"] / 24)
F["data"] = t.dt.normalize()

feri = holidays.Brazil(subdiv="MG", years=[2025, 2026])                 # nacionais + MG
facult = pd.to_datetime(["2026-02-16", "2026-02-17", "2026-06-04"])     # Carnaval (seg/ter) e Corpus Christi: ponto facultativo
fer_dates = set(pd.to_datetime(list(feri.keys())))
fac_dates = set(facult)
d = F["data"]
F["feriado"] = d.isin(fer_dates).astype(int)
F["feriado_ou_facultativo"] = (d.isin(fer_dates) | d.isin(fac_dates)).astype(int)
F["vespera_feriado"] = (d + pd.Timedelta(days=1)).isin(fer_dates).astype(int)
F["pos_feriado"] = (d - pd.Timedelta(days=1)).isin(fer_dates).astype(int)
# ponte / feriadao: feriado em seg ou sex, ou colado a fim de semana
def _prolong(day):
    if day in fer_dates or day in fac_dates:
        return int(day.dayofweek in (0, 4))
    return 0
F["feriado_prolongado"] = d.map(_prolong)
F["dist_feriado_dias"] = d.map(lambda x: min(abs((x - h).days) for h in fer_dates))
# sensibilidade de fuso/turno: "dia operacional" (madrugada pertence ao dia anterior) e fuso -3h
F["data_oper"] = (t - pd.Timedelta(hours=6)).dt.normalize()
F["feriado_oper"] = F["data_oper"].isin(fer_dates).astype(int)
F["feriado_fuso_m3"] = (t - pd.Timedelta(hours=3)).dt.normalize().isin(fer_dates).astype(int)

# ------------------------------------------------------------------ ordenar no tempo e features de contexto
F = F.sort_values(["inicio_corrida", "cd_corrida", "nu_carregamento"]).reset_index(drop=True)

# contexto da corrida: c1 -> c2
w = F.pivot(index="cd_corrida", columns="nu_carregamento",
            values=["peso_total_t", "volume_estimado_m3", "y", "inicio_cestao", "prop_pesado"])
w.columns = [f"{a}_c{b}" for a, b in w.columns]
mats = F.pivot(index="cd_corrida", columns="nu_carregamento", values="_mats")
mats.columns = ["m1", "m2"]
ctx = w.join(mats)
def _jacc(r):
    a, b = r["m1"], r["m2"]
    if not isinstance(a, frozenset) or not isinstance(b, frozenset):
        return np.nan
    return len(a & b) / len(a | b)
ctx["jaccard_materiais_12"] = ctx.apply(_jacc, axis=1)
ctx["delta_peso_12"] = ctx["peso_total_t_c2"] - ctx["peso_total_t_c1"]
ctx["razao_peso_21"] = ctx["peso_total_t_c2"] / ctx["peso_total_t_c1"]
ctx["vol_acumulado_m3"] = ctx["volume_estimado_m3_c1"] + ctx["volume_estimado_m3_c2"]
ctx["delta_prop_pesado_12"] = ctx["prop_pesado_c2"] - ctx["prop_pesado_c1"]
ctx["intervalo_c1_c2_min"] = (ctx["inicio_cestao_c2"] - ctx["inicio_cestao_c1"]).dt.total_seconds() / 60
ctx = ctx[["jaccard_materiais_12", "delta_peso_12", "razao_peso_21", "vol_acumulado_m3",
           "delta_prop_pesado_12", "intervalo_c1_c2_min", "y_c1"]].rename(columns={"y_c1": "carga_alta_c1_ctx"})
F = F.merge(ctx, on="cd_corrida", how="left")
# esses so existem p/ cestao 2 (no cestao 1 o 2 nao existe ainda) -> NaN no c1
for c in ["jaccard_materiais_12", "delta_peso_12", "razao_peso_21", "vol_acumulado_m3", "delta_prop_pesado_12",
          "intervalo_c1_c2_min", "carga_alta_c1_ctx"]:
    F.loc[F["nu_carregamento"] == 1, c] = np.nan

# historico recente (shift(1) -> nao inclui a corrida atual). Por tipo de cestao e tambem por corrida.
run = F.groupby("cd_corrida").agg(inicio=("inicio_corrida", "first"), y_run=("y_corrida", "first")).sort_values("inicio")
for N in (5, 10, 20):
    run[f"taxa_ca_ult{N}"] = run["y_run"].shift(1).rolling(N, min_periods=3).mean()
run["horas_desde_ultima_ca"] = np.nan
last = None
vals = []
for ini, y in zip(run["inicio"], run["y_run"]):
    vals.append((ini - last).total_seconds() / 3600 if last is not None else np.nan)
    if y == 1:
        last = ini
run["horas_desde_ultima_ca"] = vals
run["gap_corrida_anterior_h"] = run["inicio"].diff().dt.total_seconds() / 3600
run["corridas_desde_ultima_ca"] = np.nan
cnt, arr = None, []
for y in run["y_run"]:
    arr.append(cnt if cnt is not None else np.nan)
    cnt = 0 if y == 1 else (cnt + 1 if cnt is not None else None)
run["corridas_desde_ultima_ca"] = arr
F = F.merge(run.drop(columns=["inicio", "y_run"]), left_on="cd_corrida", right_index=True, how="left")

# interacoes com sentido fisico (guia 6.1)
F["vol_x_prop_leve"] = F["volume_estimado_m3"] * F["prop_leve"]
F["peso_x_hhi"] = F["peso_total_t"] * F["hhi_materiais"]
F["peso_leve_topo_t"] = F["peso_leve_t"] * F["prop_leve_topo"]
F["pesado_x_densidade"] = F["peso_pesado_t"] * F["densidade_media"]

# split temporal oficial (teste = ultimas 10% corridas = teste.csv)
teste = pd.read_csv(f"{SRC}/teste.csv")["cd_corrida"]
F["split"] = np.where(F["cd_corrida"].isin(teste), "teste", "dev")
dev_runs = F[F["split"] == "dev"].groupby("cd_corrida")["inicio_corrida"].first().sort_values()
cut = dev_runs.iloc[int(len(dev_runs) * 0.75)]
F.loc[(F["split"] == "dev") & (F["inicio_corrida"] >= cut), "split"] = "valid"
F.loc[F["split"] == "dev", "split"] = "treino"

F = F.drop(columns=["_mats"])
F.to_csv(f"{OUT}/features_cestao.csv", index=False)
print(F.shape, F["split"].value_counts().to_dict())
print(F.groupby(["split", "nu_carregamento"])["y"].agg(["mean", "size"]))
print("materiais frequentes:", FREQ_OK)
print("NaN por coluna (top):\n", F.isna().sum().sort_values(ascending=False).head(15))
