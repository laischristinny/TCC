import pandas as pd
import os
import numpy as np

print("--- Iniciando o pré-processamento e agregação dos dados ---")

DATA_DIR = 'dados'

CODIGO_GRUPO_COMPUTACAO = 6411

path_arq1 = os.path.join(DATA_DIR, 'microdados2023_arq1.txt')
df_cursos = pd.read_csv(
    path_arq1, sep=';', decimal='.', na_values='.', encoding='latin1'
)

cursos_computacao_ids = df_cursos[df_cursos['CO_GRUPO'] == CODIGO_GRUPO_COMPUTACAO]['CO_CURSO'].unique()

print(f"Foram encontrados {len(cursos_computacao_ids)} cursos do grupo {CODIGO_GRUPO_COMPUTACAO}.")
print(f"IDs dos cursos: {cursos_computacao_ids}\n")

df_final_agregado = pd.DataFrame({'CO_CURSO': cursos_computacao_ids}).set_index('CO_CURSO')

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
    #'microdados2023_arq5.txt': 'TP_SEXO',  # Sexo
    'microdados2023_arq8.txt': 'QE_I02', # Raça
    'microdados2023_arq14.txt': 'QE_I08',  # Renda familiar
    'microdados2023_arq23.txt': 'QE_I17',   # Tipo de Escola que concluiu o ensino médio
    'microdados2023_arq10.txt': 'QE_I04', # Escolarização do pai
    'microdados2023_arq11.txt': 'QE_I05',  # Escolarização da mãe
    'microdados2023_arq12.txt': 'QE_I06',  # Onde e com quem você mora atualmente?
    'microdados2023_arq13.txt': 'QE_I07',  # Quantas pessoas moram na sua residência?
    'microdados2023_arq15.txt': 'QE_I09',  # Renda do estudante
    'microdados2023_arq16.txt': 'QE_I10',  # Situação de trabalho
    'microdados2023_arq17.txt': 'QE_I11',  # Que tipo de bolsa de estudos ou financiamento do curso você recebeu para custear todas ou a maior parte das mensalidades?
    'microdados2023_arq18.txt': 'QE_I12',  # Você recebeu algum tipo de auxílio de permanência?
    'microdados2023_arq19.txt': 'QE_I13',  # Você recebeu algum tipo de bolsa acadêmica?
    'microdados2023_arq21.txt': 'QE_I15',  # Tipo de ingresso no curso
    'microdados2023_arq22.txt': 'QE_I16',  # Qual modalidade de ensino médio você concluiu?
    'microdados2023_arq27.txt': 'QE_I21',  # Alguém em sua família concluiu um curso superior?
    'microdados2023_arq28.txt': 'QE_I22',  # Quantos livros você leu neste ano?
    'microdados2023_arq29.txt': 'QE_I23',  # Quantas horas por semana você dedicou aos estudos?
    'microdados2023_arq31.txt': 'QE_I25',  # Qual o principal motivo para você ter escolhido este curso?
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

print("--- Tabela final agregada por curso ---")
df_final_agregado = df_final_agregado.fillna(0)


# Carregar caracterização dos cursos (arquivo 1)
df_cursos_info = pd.read_csv(
    path_arq1, sep=';', decimal='.', na_values='.', encoding='latin1'
)

# Filtrar apenas cursos de computação
df_cursos_info = df_cursos_info[df_cursos_info['CO_CURSO'].isin(cursos_computacao_ids)]

df_cursos_info = df_cursos_info.drop_duplicates(subset='CO_CURSO')

# Juntar informações (ex: IES e UF)
df_final_agregado = df_final_agregado.reset_index().merge(
    df_cursos_info[['CO_CURSO', 'CO_IES', 'CO_MODALIDADE', 'CO_UF_CURSO', 'CO_MUNIC_CURSO', 'CO_CATEGAD', 'CO_REGIAO_CURSO']],
    on='CO_CURSO',
    how='left'
).set_index('CO_CURSO')

# Salvar
df_final_agregado.to_csv('tabelas/enade_2023_computacao_agregado.csv', encoding='utf-8-sig')
print("Arquivo 'enade_2023_computacao_agregado.csv' salvo com sucesso.")
print("\nVisualização do DataFrame final:")
print(df_final_agregado.head())
