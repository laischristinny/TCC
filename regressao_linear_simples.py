import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Carregar base
df = pd.read_csv("tabelas/enade_2023_computacao_agregado.csv", encoding="utf-8-sig")

# 2. Definir target (nota) e features (explicativas)
y = df["MEDIA_NT_CE"]
X = df.drop(columns=[
    "MEDIA_NT_CE",  # variável alvo
    "CO_IES", "CO_MODALIDADE", "CO_UF_CURSO",
    "CO_MUNIC_CURSO", "CO_CATEGAD", "CO_REGIAO_CURSO"  # variáveis administrativas
])

# 3. Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Treinar modelo
lr = LinearRegression()
lr.fit(X_train, y_train)

# 5. Avaliar modelo
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)

# 6. Coeficientes
coeficientes = pd.Series(lr.coef_, index=X.columns).sort_values(ascending=False)
print("\nCoeficientes das variáveis:")
print(coeficientes.head(15))  # mostra top 15

# 7. Gráfico real vs predito
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Nota real (MEDIA_NT_CE)")
plt.ylabel("Nota predita")
plt.title("Regressão Linear - Notas Reais vs Preditas")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # linha 45°
plt.show()

"""
A regressão linear simples, quando aplicada a bases de dados que possuem elevado número de variáveis explicativas, especialmente variáveis categóricas transformadas em indicadores (dummies), tende a apresentar problemas de multicolinearidade e sobreajuste (overfitting). Tais problemas resultam em coeficientes instáveis e baixo poder preditivo, como evidenciado nos testes iniciais, em que o modelo apresentou coeficiente de determinação negativo (R² < 0), indicando pior desempenho do que a simples média das notas.
"""