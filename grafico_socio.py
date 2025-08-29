import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados agregados
try:
    df = pd.read_csv('enade_2023_computacao_agregado.csv')
except FileNotFoundError:
    print("O arquivo 'enade_2023_computacao_agregado.csv' não foi encontrado.")
else:
    # Selecionar apenas as colunas de interesse (nota e as socioeconômicas)
    colunas_interesse = ['MEDIA_NT_CE'] + [col for col in df.columns if col.startswith('QE_I08_') or col.startswith('QE_I17_')]

    df_corr = df[colunas_interesse]

    # Calcular a matriz de correlação
    matriz_correlacao = df_corr.corr()

    # Isolar as correlações com a nota média para facilitar a visualização
    corr_com_nota = matriz_correlacao[['MEDIA_NT_CE']].sort_values(by='MEDIA_NT_CE', ascending=False)


    # Gerar o heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_com_nota,
        annot=True,       # Mostrar os valores de correlação
        cmap='coolwarm',  # Esquema de cores (azul=negativo, vermelho=positivo)
        fmt=".2f",        # Formatar com duas casas decimais
        linewidths=.5
    )

    plt.title('Correlação entre Desempenho e Variáveis Socioeconômicas', fontsize=16)
    plt.tight_layout()

    # Salvar o gráfico e a tabela
    plt.savefig('grafico_heatmap_correlacao.png', dpi=300)
    print("Gráfico 'grafico_heatmap_correlacao.png' salvo com sucesso.")
    print("\nTabela de Correlação com a Média de Notas:")
    print(corr_com_nota)