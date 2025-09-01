import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Carregar a base consolidada
df = pd.read_csv("tabelas/enade_2023_computacao_agregado.csv", encoding="utf-8-sig")

# 2. Definir target (variável resposta) e features (explicativas)
y = df["MEDIA_NT_CE"]
X = df.drop(columns=[
    "MEDIA_NT_CE",  # target
    "CO_IES", "CO_MODALIDADE", "CO_UF_CURSO", "CO_MUNIC_CURSO",
    "CO_CATEGAD", "CO_REGIAO_CURSO"  # variáveis categóricas administrativas (não queremos confundir)
])

# 3. Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Treinar modelo Random Forest
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# 5. Avaliar modelo
y_pred = rf.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# 6. Importância das variáveis
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nVariáveis mais importantes para prever a nota CE:")
print(importances.head(20))  # mostra as top 20

# 7. Plotar gráfico
plt.figure(figsize=(10,6))
importances.head(20).plot(kind='bar')
plt.title("Importância das Variáveis - Random Forest")
plt.ylabel("Importância")
plt.tight_layout()
plt.show()
