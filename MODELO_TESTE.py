"""Execução demonstrativa das previsões Random Forest para os dois cestões."""
import argparse
from pathlib import Path

import pandas as pd

from modelagem.features_rf import construir_features_rf
from modelagem.prever import carregar, prever

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'modelagem' / 'saidas' / 'resultado_previsoes.csv'
MAX_EXIBIR = 20
CLASSE_CARGA_ALTA = 1


def ler_dados(caminho):
    try:
        return pd.read_csv(caminho, sep=None, engine='python', encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(caminho, sep=None, engine='python', encoding='latin1')


def adicionar_resultados(resultado, dados):
    if 't' not in resultado.columns and 'inicio_cestao' in dados.columns:
        resultado['t'] = dados.loc[resultado.index, 'inicio_cestao']

    resultado['modelo'] = resultado['nu_carregamento'].map(
        lambda numero: f'Random Forest cestão {int(numero)}'
    )
    resultado['prob_carga_alta'] = resultado['risco_rf']
    resultado['prob_carga_nao_alta'] = 1 - resultado['risco_rf']
    resultado['previsao'] = resultado['alerta'].map(
        {1: 'CARGA ALTA', 0: 'CARGA NÃO ALTA'}
    )

    alvo = 'y' if 'y' in dados.columns else 'carga_alta'
    if alvo in dados.columns:
        resultado['carga_alta_real'] = dados.loc[resultado.index, alvo]
        resultado['avaliacao'] = [
            'ACERTO'
            if (
                str(real) == str(CLASSE_CARGA_ALTA) and bool(alerta)
                or str(real) != str(CLASSE_CARGA_ALTA) and not bool(alerta)
            )
            else 'ERRO'
            for real, alerta in zip(
                resultado['carga_alta_real'],
                resultado['alerta'],
            )
        ]

    parada = 'parada_s_cestao' if 'parada_s_cestao' in dados.columns else 'parada_s'
    if parada in dados.columns:
        resultado['parada_s'] = dados.loc[resultado.index, parada]
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada', help='CSV preparado com as features RF.')
    parser.add_argument('--saida', default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    if args.entrada:
        dados = ler_dados(args.entrada)
    else:
        dados = construir_features_rf()

    cfg, modelos = carregar()
    resultado = adicionar_resultados(prever(dados, cfg=cfg, mods=modelos), dados)
    if resultado.empty:
        raise ValueError('Nenhuma previsão foi gerada.')

    caminho_saida = Path(args.saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(
        caminho_saida,
        index=False,
        encoding='utf-8-sig',
        sep=';',
    )

    print('\n' + '=' * 72)
    print(' SISTEMA DE PREVISÃO DE CARGA ALTA - RANDOM FOREST')
    print('=' * 72)
    print(f'\nBase carregada: {len(dados)} registros.')
    for numero in (1, 2):
        grupo = resultado.loc[
            pd.to_numeric(resultado['nu_carregamento'], errors='coerce').eq(numero)
        ]
        if grupo.empty:
            continue
        alertas = int(grupo['alerta'].sum())
        print(f'\nCESTÃO {numero}')
        print(f'  Registros analisados: {len(grupo)}')
        print(f'  Alertas de carga alta: {alertas}')
        print(f'  Registros sem alerta: {len(grupo) - alertas}')
        print(f'  Fração de alertas: {alertas / len(grupo):.1%}')
        print(f'  Limiar RF: {grupo["limiar_alerta"].iloc[0]:.6f}')
        if 'avaliacao' in grupo.columns:
            print('  Comparação com os rótulos disponíveis:')
            print(grupo['avaliacao'].value_counts().to_string())

    colunas_exibir = [
        coluna
        for coluna in (
            'cd_corrida',
            'nu_carregamento',
            'prob_carga_alta',
            'prob_carga_nao_alta',
            'previsao',
            'carga_alta_real',
            'avaliacao',
        )
        if coluna in resultado.columns
    ]
    exibicao = resultado[colunas_exibir].head(MAX_EXIBIR).copy()
    for coluna in ('prob_carga_alta', 'prob_carga_nao_alta'):
        if coluna in exibicao.columns:
            exibicao[coluna] = exibicao[coluna].map(lambda valor: f'{valor:.2%}')
    print('\n' + '=' * 72)
    print(f' PRIMEIROS {MAX_EXIBIR} RESULTADOS')
    print('=' * 72)
    print(exibicao.to_string(index=False))
    print('\n' + '=' * 72)
    print(' PREVISÕES CONCLUÍDAS')
    print('=' * 72)
    print(f'Arquivo salvo em:\n{caminho_saida}')


if __name__ == '__main__':
    main()
