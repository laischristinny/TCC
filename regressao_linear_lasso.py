"""
Nesse contexto, os modelos de regressão regularizada tornam-se mais adequados.

O Ridge Regression (penalização L2) introduz uma penalização que reduz a magnitude dos coeficientes, mitigando a multicolinearidade e promovendo maior estabilidade na estimação dos parâmetros.

Já o Lasso Regression (penalização L1), além de reduzir a magnitude dos coeficientes, é capaz de atribuir valor zero a variáveis menos relevantes, funcionando como um mecanismo de seleção automática de variáveis.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("tabelas/enade_2023_computacao_agregado.csv", encoding="utf-8-sig")

y = df["MEDIA_NT_CE"]
X = df.drop(columns=[
    "MEDIA_NT_CE", 
    "CO_IES", "CO_MODALIDADE", "CO_UF_CURSO",
    "CO_MUNIC_CURSO", "CO_CATEGAD", "CO_REGIAO_CURSO"
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ridge = Ridge(alpha=10)  # alpha = intensidade da regularização
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print("=== Ridge Regression ===")
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R²:", r2_score(y_test, y_pred_ridge))

coef_ridge = pd.Series(ridge.coef_, index=X.columns).sort_values(ascending=False)
print("\nCoeficientes Ridge (Top 15):")
print(coef_ridge.head(15))

lasso = Lasso(alpha=0.1)  # alpha controla o quanto zera coeficientes
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print("\n=== Lasso Regression ===")
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("R²:", r2_score(y_test, y_pred_lasso))

coef_lasso = pd.Series(lasso.coef_, index=X.columns).sort_values(ascending=False)
print("\nCoeficientes Lasso (Top 15):")
print(coef_lasso.head(15))

# 6. Gráfico comparando coeficientes Ridge vs Lasso
plt.figure(figsize=(12,6))
coef_ridge.sort_values(ascending=False).head(20).plot(kind="bar", alpha=0.6, label="Ridge")
coef_lasso.sort_values(ascending=False).head(20).plot(kind="bar", alpha=0.6, color="orange", label="Lasso")
plt.legend()
plt.title("Comparação dos coeficientes - Ridge vs Lasso")
plt.ylabel("Peso")
plt.tight_layout()
plt.show()
