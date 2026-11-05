"""Geração do conjunto de features esperado pelos Random Forests de deploy."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'dataset' / 'raw'
INTERIM = ROOT / 'dataset' / 'interim'
SCRIPTS = ROOT / 'modelagem' / 'CODIGOS_E_RESULTADOS' / 'scripts'


def construir_features_rf(raw_dir=RAW, interim_dir=INTERIM, output_path=None):
    """Constrói as features usando os scripts entregues com o novo modelo."""
    raw_dir = Path(raw_dir)
    interim_dir = Path(interim_dir)
    source_dirs = {
        'infos_camadas.csv': raw_dir,
        'dicionario_tipo_material.csv': raw_dir,
        'VariavelResposta.csv': raw_dir,
        'infos_materiais_limpo.csv': interim_dir,
        'treino.csv': interim_dir,
        'teste.csv': interim_dir,
    }

    with tempfile.TemporaryDirectory(prefix='evcomx_rf_') as temp_dir:
        temp = Path(temp_dir)
        source = temp / 'source'
        base = temp / 'base'
        result = temp / 'result'
        source.mkdir()
        base.mkdir()
        result.mkdir()

        for filename, directory in source_dirs.items():
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(f'Arquivo de entrada não encontrado: {path}')
            shutil.copy2(path, source / filename)

        feature_script = SCRIPTS / '01_features.py'
        analysis_script = SCRIPTS / 'analise_reproduzivel.py'
        subprocess.run(
            [sys.executable, str(feature_script), str(source), str(base)],
            cwd=SCRIPTS,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(analysis_script),
                str(source),
                str(base / 'features_cestao.csv'),
                str(result),
                '--features-only',
            ],
            cwd=SCRIPTS,
            check=True,
        )

        feature_path = result / 'features_modelo_rf.csv'
        features = pd.read_csv(
            feature_path,
            parse_dates=['inicio_cestao', 'inicio_corrida'],
        )

    config_path = ROOT / 'visual' / 'assets' / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    missing = {
        feature
        for info in config['cestoes'].values()
        for feature in info['features']
        if feature not in features.columns
    }
    if missing:
        raise ValueError(
            'O gerador não produziu todas as features dos modelos RF: '
            + ', '.join(sorted(missing))
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)

    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--saida',
        default=str(ROOT / 'modelagem' / 'saidas' / 'dataset_features_rf.csv'),
    )
    args = parser.parse_args()
    result = construir_features_rf(output_path=args.saida)
    print(f'{len(result)} registros gravados em {args.saida}')
