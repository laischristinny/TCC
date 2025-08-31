import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados agregados
try:
    df = pd.read_csv('tabelas/enade_2023_computacao_agregado.csv')
except FileNotFoundError:
    print("O arquivo 'enade_2023_computacao_agregado.csv' não foi encontrado.")
else:
    # ATENÇÃO: Verifique no dicionário de dados qual coluna corresponde à maior renda.
    # Vamos assumir 'QE_I08_G' como exemplo (ex: "Mais de 20 salários mínimos").
    # Se a coluna não existir, o código dará erro. Ajuste o nome da coluna abaixo.
    coluna_maior_renda = 'QE_I08_G'

    if coluna_maior_renda in df.columns:
        # Criar o gráfico de dispersão
        # 'regplot' adiciona uma linha de regressão para visualizar a tendência
        plt.figure(figsize=(10, 7))
        sns.regplot(
            x=df[coluna_maior_renda] * 100, # Multiplicar por 100 para ver como porcentagem
            y=df['MEDIA_NT_CE'],
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'}
        )

        plt.title(f'Desempenho vs. Percentual de Alunos de Alta Renda ({coluna_maior_renda})', fontsize=16)
        plt.xlabel(f'Percentual de Alunos na Faixa de Renda "{coluna_maior_renda}" (%)', fontsize=12)
        plt.ylabel('Média da Nota do Curso (NT_CE)', fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        # Salvar o gráfico
        plt.savefig('grafico_dispersao_renda_desempenho.png', dpi=300)
        print("Gráfico 'grafico_dispersao_renda_desempenho.png' salvo com sucesso.")

        # Calcular a correlação de Pearson
        correlacao = df[[coluna_maior_renda, 'MEDIA_NT_CE']].corr()
        print(f"\nCorrelação entre {coluna_maior_renda} e MEDIA_NT_CE:")
        print(correlacao)
    else:
        print(f"A coluna '{coluna_maior_renda}' não foi encontrada no DataFrame.")
        print("Verifique o nome da coluna no arquivo CSV e no dicionário de dados do ENADE.")