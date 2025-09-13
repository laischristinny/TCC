import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurações de Estilo ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

try:
    df = pd.read_csv('tabelas/enade_2023_computacao_agregado.csv')
except FileNotFoundError:
    print("Erro: O arquivo 'enade_2023_computacao_agregado.csv' não foi encontrado.")
    print("Por favor, execute o seu script de pré-processamento atualizado para incluir os dados de raça/cor.")
else:
    mapa_raca_cols = {
        'QE_I02_A': 'Branca',
        'QE_I02_B': 'Preta',
        'QE_I02_C': 'Amarela',
        'QE_I02_D': 'Parda',
        'QE_I02_E': 'Indígena',
        'QE_I02_F': 'Não declarado'
    }

    colunas_raca = [col for col in df.columns if col.startswith('QE_I02_')]
    
    if not colunas_raca:
        print("Aviso: Nenhuma coluna de raça/cor (iniciando com 'QE_I02_') foi encontrada no CSV.")
        print("Por favor, atualize e execute seu script de pré-processamento primeiro.")
    else:
        colunas_interesse = ['MEDIA_NT_CE'] + colunas_raca
        df_corr = df[colunas_interesse]

    
        corr_com_nota = df_corr.corr()['MEDIA_NT_CE'].drop('MEDIA_NT_CE')
        
        corr_com_nota.rename(index=mapa_raca_cols, inplace=True)
        corr_com_nota = corr_com_nota.sort_values(ascending=False)

        plt.figure(figsize=(12, 7))
        barplot = sns.barplot(
            x=corr_com_nota.index,
            y=corr_com_nota.values,
            palette='coolwarm_r',
            hue=corr_com_nota.index,
            dodge=False,
        )
        
        plt.axhline(0, color='black', linewidth=0.8)

        for p in barplot.patches:
            barplot.annotate(format(p.get_height(), '.2f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha = 'center', va = 'center',
                           xytext = (0, 9 if p.get_height() > 0 else -9),
                           textcoords = 'offset points')

        plt.title('Correlação entre a Nota Média do Curso e o Percentual de Alunos por Raça/Cor', fontsize=16, fontweight='bold')
        plt.xlabel('Raça/Cor Autodeclarada', fontsize=12)
        plt.ylabel('Coeficiente de Correlação de Pearson', fontsize=12)
        plt.xticks(rotation=15)
        plt.tight_layout()

        plt.savefig('imagens/grafico_correlacao_composicao_racial.png', dpi=300)
        print("\nGráfico 'grafico_correlacao_composicao_racial.png' salvo com sucesso.")

        print("\n--- Tabela: Correlação entre Nota Média e Composição Racial ---")
        print(corr_com_nota.to_frame(name='Coeficiente de Correlação'))
