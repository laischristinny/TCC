import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Carregar os dados ---
csv_file = 'tabelas/enade_2023_computacao_agregado.csv'

if not os.path.exists(csv_file):
    print(f"Erro: Arquivo '{csv_file}' não encontrado.")
else:
    df = pd.read_csv(csv_file)

    # Variável alvo
    y = df['MEDIA_NT_CE']
    # Features (remover identificadores e a própria target)
    X = df.select_dtypes(include=np.number).drop(columns=['CO_CURSO', 'MEDIA_NT_CE'])

    # --- 2. Divisão treino/teste ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 3. Treinamento do modelo ---
    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("Treinando o modelo XGBoost...")
    xgb_model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")

    # --- 4. Avaliação ---
    y_pred = xgb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Performance do XGBoost ---")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.2f}")

    # --- 5. Importância das variáveis ---
    importance = xgb_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    print("\n--- Variáveis mais importantes segundo o XGBoost ---")
    print(importance_df.head(15))

    # --- 6. Visualização: gráfico de barras ---
    plt.figure(figsize=(12, 8))
    top_vars = importance_df.head(20)  # pegar as 20 mais importantes
    sns.barplot(x="Importance", y="Feature", data=top_vars, palette="viridis")
    plt.title("Top 20 Variáveis Mais Importantes - XGBoost", fontsize=16)
    plt.xlabel("Importância (ganho de informação)", fontsize=12)
    plt.ylabel("Variável", fontsize=12)
    plt.tight_layout()

    plt.savefig("imagens/12_importancia_xgboost.png", dpi=300)
    plt.close()
    print("Gráfico salvo em 'imagens/12_importancia_xgboost.png'")
