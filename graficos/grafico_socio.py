import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados agregados
try:
    df = pd.read_csv('tabelas/enade_2023_computacao_agregado.csv')
except FileNotFoundError:
    print("O arquivo 'enade_2023_computacao_agregado.csv' não foi encontrado.")
else:
    # Selecionar apenas as colunas de interesse (nota e as socioeconômicas)
    colunas_interesse = ['MEDIA_NT_CE'] + [col for col in df.columns if col.startswith('QE_')]

    df_corr = df[colunas_interesse]

    # Calcular a matriz de correlação
    matriz_correlacao = df_corr.corr()

    # Isolar as correlações com a nota média para facilitar a visualização
    corr_com_nota = matriz_correlacao[['MEDIA_NT_CE']].sort_values(by='MEDIA_NT_CE', ascending=False)

    # Remover a correlação da própria variável (que é sempre 1)
    corr_com_nota = corr_com_nota.drop('MEDIA_NT_CE')

    # Exportar a tabela de correlação para CSV
    corr_com_nota.to_csv('tabelas/tabela_correlacao.csv')
    print("Tabela de correlação salva com sucesso em 'tabela_correlacao.csv'.")


    # Gerar o gráfico de barras horizontais
    plt.figure(figsize=(12, 18))
    sns.barplot(
        x=corr_com_nota['MEDIA_NT_CE'],
        y=corr_com_nota.index,
        orient='h' # Define a orientação do gráfico como horizontal
    )

    plt.title('Correlação entre Desempenho e Variáveis Socioeconômicas', fontsize=16)
    plt.xlabel('Correlação com a Média da Nota', fontsize=12)
    plt.ylabel('Variáveis Socioeconômicas', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6) # Adiciona uma grade para facilitar a leitura
    plt.tight_layout() # Ajusta o layout para evitar que os rótulos se sobreponham

    # Salvar o gráfico
    plt.savefig('imagens/grafico_barras_correlacao.png', dpi=300)
    print("Gráfico 'grafico_barras_correlacao.png' salvo com sucesso.")