"""P7 · Caso de negócio. Valor do minuto de parada: R$ 2.000 a 3.000 (informado pela usina).
Eficácia da ação e custo do alarme falso ainda são ASSUMIDOS: trocar pelos reais.

Ganho(q) = eficácia × minutos de parada capturados − custo_alarme × nº de falsos alarmes
Minutos capturados vêm do OOF (período de validação), anualizados proporcionalmente ao tempo coberto."""
import os, numpy as np, pandas as pd
import config as C


def rodar(eficacias=(0.3, 0.5, 0.7), custos_alarme_min=(0, 1, 3), valor_min_reais=(2000, 3000)):
    O = pd.concat([pd.read_csv(os.path.join(C.SAIDAS, f'oof_cestao{c}.csv'), parse_dates=['t']) for c in (1, 2)])
    meses = (O.t.max() - O.t.min()).days / 30.44
    total_min = O.parada_s.sum() / 60
    linhas = []
    for q in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        k = int(round(len(O) * q)); top = O.nlargest(k, 'p')
        capt = top.parada_s.sum() / 60; fa = int((top[C.ALVO] == 0).sum())
        for e in eficacias:
            for c in custos_alarme_min:
                linhas.append(dict(fracao_alerta=q, eficacia=e, custo_alarme_min=c,
                                   ganho_min_mes=(e * capt - c * fa) / meses))
    # valor do minuto de parada informado pela usina (R$ por minuto): faixa mínima e máxima
    G = pd.DataFrame(linhas)
    G['ganho_reais_mes_min'] = G.ganho_min_mes * valor_min_reais[0]
    G['ganho_reais_mes_max'] = G.ganho_min_mes * valor_min_reais[1]
    melhor = G.loc[G.groupby(['eficacia', 'custo_alarme_min']).ganho_min_mes.idxmax()].round({'ganho_min_mes': 1, 'ganho_reais_mes_min': 0, 'ganho_reais_mes_max': 0})
    G.round({'ganho_min_mes': 1, 'ganho_reais_mes_min': 0, 'ganho_reais_mes_max': 0}).to_csv(os.path.join(C.SAIDAS, 'p7_negocio_grade.csv'), index=False)
    melhor.to_csv(os.path.join(C.SAIDAS, 'p7_negocio_melhor_q.csv'), index=False)
    return dict(meses_validacao=round(meses, 1), parada_total_min_validacao=round(total_min),
                parada_min_mes=round(total_min / meses),
                custo_parada_reais_mes=(round(total_min / meses * valor_min_reais[0]), round(total_min / meses * valor_min_reais[1]))), melhor


if __name__ == '__main__':
    info, melhor = rodar(); print(info); print(melhor.to_string(index=False))
