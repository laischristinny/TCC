import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt


"""
H₀: MEDIA_NT_CE não difere entre tipos de escola predominantes (privada, publica...).

H₁: pelo menos um tipo difere.
"""

df_final = pd.read_csv('tabelas/enade_2023_computacao_agregado.csv')

faixas_renda = ['QE_I17_A','QE_I17_B','QE_I17_C','QE_I17_D','QE_I17_E','QE_I17_F']

df_final['faixa_tipo_escola_predom'] = df_final[faixas_renda].idxmax(axis=1)

# --- ANOVA ---
grupos = [df_final[df_final['faixa_tipo_escola_predom']==cat]['MEDIA_NT_CE']
          for cat in df_final['faixa_tipo_escola_predom'].unique()]

f_stat, p_val = stats.f_oneway(*grupos)
print("ANOVA:", f_stat, p_val)

# --- Teste Tukey ---
tukey = pairwise_tukeyhsd(endog=df_final['MEDIA_NT_CE'],
                          groups=df_final['faixa_tipo_escola_predom'],
                          alpha=0.05)

print("\nResultado do teste de Tukey:")
print(tukey.summary())

mc = MultiComparison(df_final['MEDIA_NT_CE'], df_final['faixa_tipo_escola_predom'])
tukey_result = mc.tukeyhsd()

fig, ax = plt.subplots(figsize=(8, 6))
tukey_result.plot_simultaneous(ax=ax)
plt.title("Intervalos de Confiança - Teste de Tukey")
plt.xlabel("Diferença de Média")
plt.grid(True)
plt.show()


"""
ANOVA: 18.106828068074797 3.757760054139824e-05

Resultado do teste de Tukey:
 Multiple Comparison of Means - Tukey HSD, FWER=0.05
======================================================
 group1   group2  meandiff p-adj lower   upper  reject
------------------------------------------------------
QE_I17_A QE_I17_B  10.2541   0.0 5.4907 15.0175   True
------------------------------------------------------
"""