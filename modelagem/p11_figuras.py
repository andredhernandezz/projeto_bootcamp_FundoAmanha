"""P11 · Figuras para o relatório/apresentação (Aula IX). Usa as previsões fora-da-amostra do dev (saidas/)."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as C, p6_calibracao_limiar as p6

AZ, LA = '#2a6f97', '#c8553d'


def limpa(ax):
    ax.spines[['top', 'right']].set_visible(False); ax.grid(alpha=.25)


def rodar():
    fd = os.path.join(C.SAIDAS, 'figuras'); os.makedirs(fd, exist_ok=True)
    O = pd.concat([pd.read_csv(os.path.join(C.SAIDAS, f'oof_cestao{c}.csv')) for c in (1, 2)])
    O = O.sort_values('p', ascending=False).reset_index(drop=True)
    n = len(O); x = np.arange(1, n + 1) / n
    capt = O.parada_s.cumsum() / O.parada_s.sum(); rec = O[C.ALVO].cumsum() / O[C.ALVO].sum()
    prec = O[C.ALVO].cumsum() / np.arange(1, n + 1)
    # 1. captura de minutos
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(x, capt, color=AZ, lw=2.2, label='minutos de parada capturados'); ax.plot(x, rec, color=LA, lw=1.8, ls='--', label='cargas altas capturadas (recall)')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, ls=':', label='ao acaso')
    ax.axvline(.2, color='k', lw=.8, alpha=.5); i = int(.2 * n) - 1
    ax.annotate(f'20% sinalizados:\n{capt[i]:.0%} dos minutos', (.2, capt[i]), (.33, capt[i] - .18), arrowprops=dict(arrowstyle='-', color='k'), fontsize=9)
    ax.set_xlabel('fração dos cestões sinalizados (maior risco primeiro)'); ax.set_ylabel('fração capturada'); ax.legend(frameon=False, fontsize=8)
    ax.set_title('Quanto da parada o alerta alcança'); limpa(ax); plt.tight_layout(); plt.savefig(os.path.join(fd, 'captura_minutos.png'), dpi=150); plt.close()
    # 2. precisão x fração
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(x[20:], prec[20:], color=AZ, lw=2.2); ax.axhline(O[C.ALVO].mean(), color='gray', ls=':', lw=1, label=f'taxa geral ({O[C.ALVO].mean():.0%})')
    ax.set_xlabel('fração dos cestões sinalizados'); ax.set_ylabel('precisão (acertos entre os alertas)'); ax.legend(frameon=False, fontsize=8)
    ax.set_title('Mais alertas, menos precisão'); limpa(ax); plt.tight_layout(); plt.savefig(os.path.join(fd, 'precisao_vs_alertas.png'), dpi=150); plt.close()
    # 3. calibração
    _, _, rel, _ = p6.rodar()
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, .6], [0, .6], color='gray', ls=':'); ax.plot(rel.p_media, rel.taxa_real, 'o-', color=AZ)
    ax.set_xlabel('risco previsto (calibrado)'); ax.set_ylabel('taxa real de carga alta'); ax.set_title('Calibração (metade mais recente)'); limpa(ax)
    plt.tight_layout(); plt.savefig(os.path.join(fd, 'calibracao.png'), dpi=150); plt.close()
    print('figuras em', fd)


if __name__ == '__main__':
    rodar()
