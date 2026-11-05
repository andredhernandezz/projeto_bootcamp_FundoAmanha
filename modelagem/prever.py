"""Inferência Random Forest para risco de carga alta por cestão.

Entrada: CSV com as features RF geradas por features_rf.py, ou geração automática
         a partir dos arquivos em dataset/raw/ e dataset/interim/.
Saída: CSV com risco_rf (probabilidade bruta), limiar_alerta e alerta.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__:
    from .features_rf import construir_features_rf
else:
    from features_rf import construir_features_rf

ROOT = Path(__file__).resolve().parents[1]
PASTA_LOCAL = Path(__file__).resolve().parent / 'modelo_final'
PASTA_PACOTE = ROOT / 'visual' / 'assets'


def _pasta_modelos():
    configurada = os.environ.get('EVCOMX_RF_MODEL_DIR')
    if configurada:
        return Path(configurada)

    modelos_locais = [
        PASTA_LOCAL / 'random_forest_c1.joblib',
        PASTA_LOCAL / 'random_forest_c2.joblib',
        PASTA_LOCAL / 'random_forest_config.json',
    ]
    if all(path.is_file() for path in modelos_locais):
        return PASTA_LOCAL
    if any(path.exists() for path in modelos_locais):
        raise FileNotFoundError(
            f'Conjunto local de modelos incompleto em {PASTA_LOCAL}. '
            'Execute treinar_final.py novamente.'
        )

    warnings.warn(
        f'Usando os artefatos RF validados do pacote visual em {PASTA_PACOTE}; '
        'execute treinar_final.py para criar um conjunto local.',
        RuntimeWarning,
        stacklevel=2,
    )
    return PASTA_PACOTE


def carregar(model_dir=None):
    pasta = Path(model_dir) if model_dir is not None else _pasta_modelos()
    nome_config = (
        'random_forest_config.json'
        if (pasta / 'random_forest_config.json').is_file()
        else 'config.json'
    )
    config_path = pasta / nome_config
    if not config_path.is_file():
        raise FileNotFoundError(f'Configuração dos modelos não encontrada: {config_path}')

    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    modelos = {}
    for c in cfg['cestoes']:
        model_path = pasta / f'random_forest_c{c}.joblib'
        if not model_path.is_file():
            raise FileNotFoundError(f'Modelo do cestão {c} não encontrado: {model_path}')
        modelos[c] = joblib.load(model_path)
    return cfg, modelos


def prever(df, q=None, cfg=None, mods=None):
    """Prevê usando os limiares RF validados; ``q`` opcional define corte por quantil."""
    if (cfg is None) != (mods is None):
        raise ValueError('Informe cfg e mods juntos ou deixe ambos vazios para carregá-los.')
    if cfg is None:
        cfg, mods = carregar()
    if q is not None and not 0 < q < 1:
        raise ValueError('q precisa estar estritamente entre 0 e 1.')
    if 'nu_carregamento' not in df.columns:
        raise ValueError('A entrada precisa conter a coluna nu_carregamento.')

    cestao = pd.to_numeric(df['nu_carregamento'], errors='coerce')
    resultados = []
    for c, info in cfg['cestoes'].items():
        numero = int(c)
        x = df.loc[cestao.eq(numero)]
        if x.empty:
            continue

        features = info['features']
        faltantes = [feature for feature in features if feature not in x.columns]
        if faltantes:
            raise ValueError(
                f'Faltam features RF do cestão {numero}: {", ".join(faltantes)}'
            )

        xx = x.loc[:, features].copy()
        for coluna in features:
            valores = xx[coluna]
            if not pd.api.types.is_numeric_dtype(valores):
                valores = valores.astype('string').str.replace(',', '.', regex=False)
            xx[coluna] = pd.to_numeric(valores, errors='raise')
        xx = xx.replace([np.inf, -np.inf], np.nan)

        if numero == 2:
            contexto = xx['carga_alta_c1_ctx']
            if contexto.isna().any() or not contexto.isin([0, 1]).all():
                raise ValueError(
                    'Para o cestão 2, carga_alta_c1_ctx deve ser o resultado '
                    'observado (0 ou 1) do cestão 1.'
                )

        probabilidade = mods[c].predict_proba(xx)[:, 1]
        limiar = (
            float(np.quantile(probabilidade, 1 - q))
            if q is not None
            else float(info['limiar'])
        )
        colunas_saida = [
            coluna
            for coluna in ('cd_corrida', 'nu_carregamento', 't', 'inicio_cestao')
            if coluna in x.columns
        ]
        resultado = x.loc[:, colunas_saida].copy()
        resultado['risco_rf'] = probabilidade
        resultado['limiar_alerta'] = limiar
        resultado['alerta'] = (probabilidade >= limiar).astype(int)
        resultados.append(resultado)

    if not resultados:
        raise ValueError('Nenhuma previsão foi gerada para os cestões 1 ou 2.')
    return pd.concat(resultados).sort_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada', help='CSV já preparado com as features RF.')
    parser.add_argument('--saida', default='previsoes.csv')
    parser.add_argument(
        '--q',
        type=float,
        default=None,
        help='Opcional: usa o quantil informado em vez dos limiares RF validados.',
    )
    parser.add_argument('--ultimos', type=int, default=0)
    parser.add_argument('--dados-raw', default=str(ROOT / 'dataset' / 'raw'))
    parser.add_argument('--dados-interim', default=str(ROOT / 'dataset' / 'interim'))
    args = parser.parse_args()

    if args.entrada:
        df = pd.read_csv(args.entrada, sep=None, engine='python', encoding='utf-8-sig')
    else:
        df = construir_features_rf(args.dados_raw, args.dados_interim)
    if args.ultimos:
        df = df.tail(args.ultimos)

    resultado = prever(df, q=args.q)
    resultado.to_csv(args.saida, index=False, encoding='utf-8-sig')
    print(
        f'{len(resultado)} cestões | alertas: {int(resultado.alerta.sum())} '
        f'({resultado.alerta.mean():.1%}) | risco RF médio {resultado.risco_rf.mean():.3f}'
    )
    print(resultado.tail(8).to_string(index=False))


if __name__ == '__main__':
    main()
