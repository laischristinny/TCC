import pandas as pd
import os
import numpy as np

print("--- Iniciando o pré-processamento e agregação dos dados ---")

# Caminho da pasta onde estão os arquivos .txt
DATA_DIR = 'dados'

# 1. FILTRAGEM: Identificar os cursos de Computação
CODIGO_GRUPO_COMPUTACAO = 6411

path_arq1 = os.path.join(DATA_DIR, 'microdados2023_arq1.txt')
df_cursos = pd.read_csv(
    path_arq1, sep=';', decimal='.', na_values='.', encoding='latin1'
)

cursos_computacao_ids = df_cursos[df_cursos['CO_GRUPO'] == CODIGO_GRUPO_COMPUTACAO]['CO_CURSO'].unique()

print(f"Foram encontrados {len(cursos_computacao_ids)} cursos do grupo {CODIGO_GRUPO_COMPUTACAO}.")
print(f"IDs dos cursos: {cursos_computacao_ids}\n")

# 2. AGREGAÇÃO: Criar DataFrame final
df_final_agregado = pd.DataFrame({'CO_CURSO': cursos_computacao_ids}).set_index('CO_CURSO')

# 3. Processar DESEMPENHO
print("Processando notas do componente específico (NT_CE)...")
path_arq3 = os.path.join(DATA_DIR, 'microdados2023_arq3.txt')
df_notas = pd.read_csv(
    path_arq3, sep=';', decimal='.', na_values='.', encoding='latin1'
)

# Filtrar cursos de computação
df_notas = df_notas[df_notas['CO_CURSO'].isin(cursos_computacao_ids)]

# Converter NT_CE para numérico ('.' vira NaN automaticamente por na_values)
df_notas['NT_CE'] = pd.to_numeric(df_notas['NT_CE'], errors='coerce')

# Média por curso
media_notas_por_curso = df_notas.groupby('CO_CURSO')['NT_CE'].mean()
df_final_agregado = df_final_agregado.join(media_notas_por_curso.rename('MEDIA_NT_CE'))
print("Média de notas por curso calculada.\n")

# 4. Processar SOCIOECONÔMICAS
arquivos_categoricos = {
    'microdados2023_arq8.txt': 'QE_I02', # Raça
    'microdados2023_arq14.txt': 'QE_I08',  # Renda (confirme no dicionário)
    'microdados2023_arq23.txt': 'QE_I17'   # Tipo de Escola (confirme no dicionário)
}

for filename, variavel in arquivos_categoricos.items():
    print(f"Processando variável '{variavel}' do arquivo '{filename}'...")
    path_arquivo = os.path.join(DATA_DIR, filename)
    df_temp = pd.read_csv(
        path_arquivo, sep=';', decimal='.', na_values='.', encoding='latin1'
    )

    # Filtrar cursos de computação e remover ausentes
    df_temp = df_temp[df_temp['CO_CURSO'].isin(cursos_computacao_ids)]
    df_temp = df_temp[df_temp[variavel].notna()]

    # Distribuição percentual
    distribuicao_percentual = (
        df_temp.groupby('CO_CURSO')[variavel]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    distribuicao_percentual = distribuicao_percentual.add_prefix(f'{variavel}_')

    df_final_agregado = df_final_agregado.join(distribuicao_percentual)
    print(f"Distribuição percentual da variável '{variavel}' agregada.\n")

# 5. RESULTADO FINAL
print("--- Tabela final agregada por curso ---")
df_final_agregado = df_final_agregado.fillna(0)


# Carregar caracterização dos cursos (arquivo 1)
df_cursos_info = pd.read_csv(
    path_arq1, sep=';', decimal='.', na_values='.', encoding='latin1'
)

# Filtrar apenas cursos de computação
df_cursos_info = df_cursos_info[df_cursos_info['CO_CURSO'].isin(cursos_computacao_ids)]

# Juntar informações (ex: IES e UF)
df_final_agregado = df_final_agregado.reset_index().merge(
    df_cursos_info[['CO_CURSO', 'CO_IES', 'CO_MODALIDADE', 'CO_UF_CURSO', 'CO_MUNIC_CURSO']],
    on='CO_CURSO',
    how='left'
).set_index('CO_CURSO')

# Salvar
df_final_agregado.to_csv('enade_2023_computacao_agregado.csv', encoding='utf-8-sig')
print("Arquivo 'enade_2023_computacao_agregado.csv' salvo com sucesso.")
print("\nVisualização do DataFrame final:")
print(df_final_agregado.head())
