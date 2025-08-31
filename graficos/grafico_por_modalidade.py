import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados agregados
try:
    df = pd.read_csv('tabelas/enade_2023_computacao_agregado.csv')
except FileNotFoundError:
    print("O arquivo 'enade_2023_computacao_agregado.csv' não foi encontrado.")
else:
    # Mapear códigos de modalidade para nomes
    df['MODALIDADE'] = df['CO_MODALIDADE'].map({1: 'Presencial', 0: 'EAD'})

    # Gerar o gráfico
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='MODALIDADE', y='MEDIA_NT_CE', data=df, palette='pastel')

    plt.title('Distribuição das Notas por Modalidade de Ensino', fontsize=16)
    plt.xlabel('Modalidade', fontsize=12)
    plt.ylabel('Média da Nota do Curso (NT_CE)', fontsize=12)
    plt.tight_layout()

    # Salvar o gráfico
    plt.savefig('grafico_boxplot_modalidade.png', dpi=300)
    print("Gráfico 'grafico_boxplot_modalidade.png' salvo com sucesso.")

    # Tabela de estatísticas descritivas
    tabela_descritiva = df.groupby('MODALIDADE')['MEDIA_NT_CE'].describe()
    print("\nTabela: Estatísticas Descritivas por Modalidade")
    print(tabela_descritiva)