import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt


"""
Hipótese nula (H₀): não há diferença no desempenho médio (MEDIA_NT_CE) 
entre cursos com diferentes faixas de renda dos alunos.

Hipótese alternativa (H₁): pelo menos uma faixa de renda está associada 
a desempenho médio diferente.
"""

df_final = pd.read_csv('tabelas/enade_2023_computacao_agregado.csv')

# Colunas das faixas de renda
faixas_renda = ['QE_I08_A','QE_I08_B','QE_I08_C','QE_I08_D','QE_I08_E','QE_I08_F','QE_I08_G']

df_final['faixa_renda_predom'] = df_final[faixas_renda].idxmax(axis=1)

# --- ANOVA ---
grupos = [df_final[df_final['faixa_renda_predom']==cat]['MEDIA_NT_CE']
          for cat in df_final['faixa_renda_predom'].unique()]

f_stat, p_val = stats.f_oneway(*grupos)
print("ANOVA:", f_stat, p_val)

# --- Teste Tukey ---
tukey = pairwise_tukeyhsd(endog=df_final['MEDIA_NT_CE'],
                          groups=df_final['faixa_renda_predom'],
                          alpha=0.05)

print("\nResultado do teste de Tukey:")
print(tukey.summary())

mc = MultiComparison(df_final['MEDIA_NT_CE'], df_final['faixa_renda_predom'])
tukey_result = mc.tukeyhsd()

fig, ax = plt.subplots(figsize=(8, 6))
tukey_result.plot_simultaneous(ax=ax)
plt.title("Intervalos de Confiança - Teste de Tukey")
plt.xlabel("Diferença de Média")
plt.grid(True)
plt.show()

""" 
ANOVA: 13.28349755720695 1.3990391493501314e-10

Resultado do teste de Tukey:
   Multiple Comparison of Means - Tukey HSD, FWER=0.05   
=========================================================
 group1   group2  meandiff p-adj   lower    upper  reject
---------------------------------------------------------
QE_I08_A QE_I08_B  14.7049    0.0   6.0513 23.3584   True
QE_I08_A QE_I08_C  17.6188    0.0   9.1643 26.0734   True
QE_I08_A QE_I08_D  12.3186   0.02   1.2332  23.404   True
QE_I08_A QE_I08_E  21.6518    0.0  11.6595  31.644   True
QE_I08_A QE_I08_F  28.1053    0.0  16.7489 39.4616   True
QE_I08_B QE_I08_C    2.914 0.8191  -3.9065  9.7345  False
QE_I08_B QE_I08_D  -2.3863  0.982  -12.282  7.5094  False
QE_I08_B QE_I08_E   6.9469 0.1931  -1.7066 15.6004  False
QE_I08_B QE_I08_F  13.4004 0.0029   3.2021 23.5987   True
QE_I08_C QE_I08_D  -5.3003  0.616 -15.0225  4.4219  False
QE_I08_C QE_I08_E   4.0329 0.7398  -4.4217 12.4875  False
QE_I08_C QE_I08_F  10.4864 0.0348   0.4564 20.5164   True
QE_I08_D QE_I08_E   9.3332 0.1522  -1.7522 20.4186  False
QE_I08_D QE_I08_F  15.7867 0.0041   3.4576 28.1158   True
QE_I08_E QE_I08_F   6.4535 0.5722  -4.9028 17.8098  False
---------------------------------------------------------
"""