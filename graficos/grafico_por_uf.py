import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados agregados
try:
    df = pd.read_csv('tabelas/enade_2023_computacao_agregado.csv')
except FileNotFoundError:
    print("O arquivo 'enade_2023_computacao_agregado.csv' não foi encontrado.")
    print("Certifique-se de executar o script de pré-processamento primeiro.")
else:
    # Mapear códigos de UF para nomes para melhor visualização
    mapa_uf = {
        11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
        21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA',
        31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
        41: 'PR', 42: 'SC', 43: 'RS',
        50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
    }
    df['SG_UF'] = df['CO_UF_CURSO'].map(mapa_uf)

    # Calcular a média da nota (NT_CE) por estado
    media_por_uf = df.groupby('SG_UF')['MEDIA_NT_CE'].mean().sort_values(ascending=False)

    # Gerar o gráfico
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))
    sns.barplot(x=media_por_uf.index, y=media_por_uf.values, palette='viridis')

    plt.title('Média da Nota do Componente Específico (NT_CE) por Estado', fontsize=16)
    plt.xlabel('Estado (UF)', fontsize=12)
    plt.ylabel('Média da Nota', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Salvar o gráfico
    plt.savefig('grafico_media_nota_por_estado.png', dpi=300)

    print("Gráfico 'grafico_media_nota_por_estado.png' salvo com sucesso.")
    print("\nTabela: Média de Notas por Estado")
    print(media_por_uf.to_frame())