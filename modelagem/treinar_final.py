"""Treina os dois Random Forests com o split oficial de treino."""
import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

if __package__:
    from .features_rf import construir_features_rf
else:
    from features_rf import construir_features_rf

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(__file__).resolve().parent / 'modelo_final'
REFERENCE_CONFIG = ROOT / 'visual' / 'assets' / 'config.json'


def treinar(output_dir=MODEL_DIR):
    features = construir_features_rf()
    config = json.loads(REFERENCE_CONFIG.read_text(encoding='utf-8'))
    treino = features.loc[features['split'].eq('treino')]
    if treino.empty:
        raise ValueError('O gerador RF não produziu linhas do split treino.')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = config['hiperparametros']
    for c in ('1', '2'):
        info = config['cestoes'][c]
        dados_cestao = treino.loc[treino['nu_carregamento'].eq(int(c))]
        y = dados_cestao['y'].astype(int)
        if y.nunique() != 2:
            raise ValueError(f'O treino do cestão {c} não contém as duas classes.')

        model = make_pipeline(
            SimpleImputer(strategy='median'),
            RandomForestClassifier(
                n_estimators=params['n_estimators'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=params['random_state'],
                n_jobs=-1,
            ),
        )
        model.fit(dados_cestao[info['features']], y)
        path = output_dir / f'random_forest_c{c}.joblib'
        joblib.dump(model, path)
        print(
            f'Cestão {c}: treinado em {len(dados_cestao)} linhas; '
            f'{len(info["features"])} features; salvo em {path}'
        )

    config['treinado_com_holdout'] = False
    config['split_treino'] = 'treino'
    config_path = output_dir / 'random_forest_config.json'
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'Configuração salva em {config_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saida', default=str(MODEL_DIR))
    args = parser.parse_args()
    treinar(args.saida)
