import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("tabelas/enade_2023_computacao_agregado.csv", encoding="utf-8-sig")

y = df["MEDIA_NT_CE"]
X = df.drop(columns=[
    "MEDIA_NT_CE",
    "CO_IES", "CO_MODALIDADE", "CO_UF_CURSO", "CO_MUNIC_CURSO",
    "CO_CATEGAD", "CO_REGIAO_CURSO"
])

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar modelo Random Forest
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Avaliar modelo
y_pred = rf.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Importância das variáveis
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nVariáveis mais importantes para prever a nota CE:")
print(importances.head(20))

# Plotar gráfico
plt.figure(figsize=(10,6))
importances.head(20).plot(kind='bar')
plt.title("Importância das Variáveis - Random Forest")
plt.ylabel("Importância")
plt.tight_layout()
plt.show()

"""
MSE: 31.827468584928244
R²: 0.6483718491837729

Variáveis mais importantes para prever a nota CE:
QE_I13_B    0.186998
QE_I25_E    0.091766
QE_I17_B    0.091271
QE_I09_E    0.086404
QE_I04_D    0.061775
QE_I21_A    0.044631
QE_I15_A    0.038376
"""