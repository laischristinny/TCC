import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Carregar base
df = pd.read_csv("tabelas/enade_2023_computacao_agregado.csv", encoding="utf-8-sig")

# 2. Definir target e features
y = df["MEDIA_NT_CE"]
X = df.drop(columns=[
    "MEDIA_NT_CE",  
    "CO_IES", "CO_MODALIDADE", "CO_UF_CURSO", "CO_MUNIC_CURSO",
    "CO_CATEGAD", "CO_REGIAO_CURSO"
])

# 3. Definir modelo e KFold
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 4. Guardar métricas e importâncias
r2_scores, mse_scores = [], []
importances_list = []

for train_idx, test_idx in cv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # métricas
    r2_scores.append(r2_score(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))

    # importância das variáveis
    importances_list.append(rf.feature_importances_)

# 5. Resultados médios
print("Resultados Validação Cruzada (5 folds):")
print("R² médio:", np.mean(r2_scores))
print("R² por fold:", r2_scores)
print("MSE médio:", np.mean(mse_scores))
print("MSE por fold:", mse_scores)

# 6. Importância média das variáveis
importances_mean = np.mean(importances_list, axis=0)
importances_std = np.std(importances_list, axis=0)

importances_df = pd.DataFrame({
    "Variável": X.columns,
    "Importância Média": importances_mean,
    "Desvio-Padrão": importances_std
}).sort_values("Importância Média", ascending=False)

print("\nTop 20 variáveis mais importantes (média nos folds):")
print(importances_df.head(20))

# 7. Plotar gráfico
plt.figure(figsize=(10,6))
plt.bar(importances_df["Variável"].head(20),
        importances_df["Importância Média"].head(20),
        yerr=importances_df["Desvio-Padrão"].head(20),
        capsize=4)
plt.title("Importância das Variáveis - Random Forest (Validação Cruzada)")
plt.ylabel("Importância média (± desvio-padrão)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
