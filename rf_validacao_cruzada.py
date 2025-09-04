import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
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

rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores, mse_scores = [], []
importances_list = []

for train_idx, test_idx in cv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    r2_scores.append(r2_score(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))

    importances_list.append(rf.feature_importances_)

print("Resultados Validação Cruzada (5 folds):")
print("R² médio:", np.mean(r2_scores))
print("R² por fold:", r2_scores)
print("MSE médio:", np.mean(mse_scores))
print("MSE por fold:", mse_scores)

importances_mean = np.mean(importances_list, axis=0)
importances_std = np.std(importances_list, axis=0)

importances_df = pd.DataFrame({
    "Variável": X.columns,
    "Importância Média": importances_mean,
    "Desvio-Padrão": importances_std
}).sort_values("Importância Média", ascending=False)

print("\nTop 20 variáveis mais importantes (média nos folds):")
print(importances_df.head(20))

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

"""
Resultados Validação Cruzada (5 folds):
R² médio: 0.724683368475256
R² por fold: [0.6418484580187995, 0.8399129503979381, 0.6646674901610543, 0.7486726710565481, 0.7283152727419406]
MSE médio: 40.81669226992305
MSE por fold: [32.417930488756014, 29.27599880202273, 41.44093015062865, 42.96699093766261, 57.98161097054524]

Top 20 variáveis mais importantes (média nos folds):
     Variável  Importância Média  Desvio-Padrão
75   QE_I13_B           0.161659       0.045128
130  QE_I25_E           0.102091       0.057284
114  QE_I21_A           0.060338       0.028173
23   QE_I04_D           0.052443       0.025950
15   QE_I17_B           0.049073       0.030719
123  QE_I23_C           0.044283       0.024984
122  QE_I23_B           0.044057       0.048571
80   QE_I15_A           0.037665       0.017651
74   QE_I13_A           0.034488       0.019550
50   QE_I09_E           0.029983       0.030216
12   QE_I08_F           0.027577       0.008659
"""