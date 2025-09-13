import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)

coeficientes = pd.Series(lr.coef_, index=X.columns).sort_values(ascending=False)
print("\nCoeficientes das variáveis:")
print(coeficientes.head(15))

plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Nota real (MEDIA_NT_CE)")
plt.ylabel("Nota predita")
plt.title("Regressão Linear - Notas Reais vs Preditas")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()


"""
MSE: 57.13780664315817
R²: 0.6090006069110576

Coeficientes das variáveis:
QE_I17_F    19.476634
QE_I17_C    15.984207
QE_I06_E    14.120501
QE_I07_A    10.255554
QE_I08_F    10.101124
"""
"""
A regressão linear simples, quando aplicada a bases de dados que possuem elevado número de variáveis explicativas, especialmente variáveis categóricas transformadas em indicadores (dummies), tende a apresentar problemas de multicolinearidade e sobreajuste (overfitting). Tais problemas resultam em coeficientes instáveis e baixo poder preditivo, como evidenciado nos testes iniciais, em que o modelo apresentou coeficiente de determinação negativo (R² < 0), indicando pior desempenho do que a simples média das notas.
"""

