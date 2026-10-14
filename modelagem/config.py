"""Configuração central da modelagem. Mude aqui, não nos módulos."""
import os
DADOS = os.environ.get('DADOS_DIR', '.')      # pasta com os CSVs da EDA
SAIDAS = os.environ.get('SAIDAS_DIR', 'saidas')
os.makedirs(SAIDAS, exist_ok=True)
SEED = 42
ALVO = 'carga_alta'
N_SPLITS = 5          # folds da validação temporal
FRACAO_ALERTA = 0.20  # fração de cestões sinalizados nas métricas de operação

# Colunas conhecidas ANTES de carregar (ver EDA, parte 9)
BASE = ['peso_total_t', 'num_camadas', 'num_materiais', 'camadas_peso_zero',
        'volume_estimado_m3', 'peso_leve_t', 'peso_misto_t', 'peso_pesado_t',
        'prop_leve', 'prop_misto', 'prop_pesado', 'densidade_aparente',
        'base_leve', 'topo_leve', 'leve_no_meio_pct', 'pesado_fora_meio_pct', 'regra_ok']
TEMPO = ['hora', 'dia_semana']

# Colunas PROIBIDAS como feature (pós-evento ou derivadas do alvo)
PROIBIDAS = ['tem_gusa_liquido', 'peso_carreg3_t', 'parada_s', 'is_carga_alta', 'carga_alta']
