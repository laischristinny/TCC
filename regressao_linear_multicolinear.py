import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. PREPARAÇÃO DOS DADOS ---

# Configurações de estilo para os gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Nome do arquivo CSV gerado pelo script de pré-processamento
# ATENÇÃO: Verifique se este é o nome correto do seu arquivo
csv_file = 'tabelas/enade_2023_computacao_agregado.csv'

if not os.path.exists(csv_file):
    print(f"Erro: Arquivo '{csv_file}' não encontrado.")
    print("Execute o script de pré-processamento primeiro para gerar este arquivo.")
else:
    # Carregar os dados agregados
    df = pd.read_csv(csv_file)
    # Remove as linhas duplicadas
    # df.drop_duplicates(inplace=True)
    # print(f"Formato após remover duplicatas: {df.shape}\n")
    # Definir a variável alvo (y) e as preditoras (X)
    y = df['MEDIA_NT_CE']
    # Usar todas as colunas numéricas como features, exceto os identificadores e a própria variável alvo
    features = df.select_dtypes(include=np.number).drop(columns=['CO_CURSO', 'MEDIA_NT_CE'])
    X = features
    
    feature_names = X.columns

    print("--- Dados carregados e prontos para a modelagem ---")

    # --- 2. DIVISÃO DOS DADOS EM TREINO E TESTE ---
    # Usamos o mesmo random_state=42 para garantir que a divisão seja a mesma,
    # permitindo uma comparação justa entre os modelos.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dados divididos: {len(X_train)} amostras de treino, {len(X_test)} amostras de teste.")

    # --- 3. TREINAMENTO DO MODELO DE REGRESSÃO LINEAR ---
    lr_model = LinearRegression()

    print("\nTreinando o modelo de Regressão Linear...")
    lr_model.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")

    # --- 4. AVALIAÇÃO DO MODELO ---
    y_pred_lr = lr_model.predict(X_test)

    # Calcular métricas de erro
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print("\n--- Performance do Modelo no Conjunto de Teste ---")
    print(f"Erro Quadrático Médio (MSE): {mse_lr:.2f}")
    print(f"Coeficiente de Determinação (R²): {r2_lr:.2f}")
    print(f"(Compare este R² com o do Random Forest para ver qual modelo explica melhor os dados)")

    # --- 5. ANÁLISE DOS COEFICIENTES ---
    # Esta é a principal análise para este modelo
    coefficients = lr_model.coef_
    coeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False)

    print("\n--- Coeficientes do Modelo de Regressão Linear ---")
    print(coeff_df)

    # --- 6. VISUALIZAÇÃO DOS RESULTADOS ---
    
    # Criar a pasta para salvar os gráficos, se não existir
    if not os.path.exists('graficos'):
        os.makedirs('graficos')

    # Gráfico 10: Coeficientes da Regressão Linear
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coeff_df, palette='vlag')
    plt.title('Gráfico 10: Coeficientes da Regressão Linear para cada Fator', fontsize=16)
    plt.xlabel('Valor do Coeficiente', fontsize=12)
    plt.ylabel('Fatores Socioeconômicos e do Curso', fontsize=12)
    plt.axvline(x=0, color='black', linewidth=0.8) # Linha no zero para referência
    plt.tight_layout()
    plt.savefig('graficos/10_coeficientes_regressao_linear.png', dpi=300)
    plt.close()
    print("\nGráfico 10 (Coeficientes da Regressão) salvo com sucesso.")

    # Gráfico 11: Valores Reais vs. Previstos (para Regressão Linear)
    plt.figure()
    plt.scatter(y_test, y_pred_lr, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.xlabel('Notas Reais', fontsize=12)
    plt.ylabel('Notas Previstas (Regressão Linear)', fontsize=12)
    plt.title('Gráfico 11: Comparação entre Notas Reais e Previstas (Regressão Linear)', fontsize=16)
    plt.tight_layout()
    plt.savefig('graficos/11_real_vs_previsto_linear.png', dpi=300)
    plt.close()
    print("Gráfico 11 (Real vs. Previsto da Regressão Linear) salvo com sucesso.")

    print("\nFase de modelagem com Regressão Linear concluída.")

"""
Reporte o Resultado da Regressão Linear Exatamente Como Encontrou: 
Apresente o R² de 0.99 e os coeficientes gigantes.

Discuta a Multicolinearidade: Explique no seu texto que a alta performance do modelo, combinada com os coeficientes instáveis, é um forte indicativo de multicolinearidade. Explique que isso ocorre porque as colunas de percentuais para uma mesma pergunta do questionário (como Cor/Raça) são perfeitamente dependentes entre si.

Use o Random Forest Como a Solução: Este é o momento perfeito para introduzir o Random Forest. Modelos baseados em árvores (como o Random Forest) não são afetados pela multicolinearidade da mesma forma que a Regressão Linear. Portanto, a análise de importância das features do Random Forest será muito mais confiável e interpretável.
"""