import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt


"""
H₀: MEDIA_NT_CE não difere entre cursos com diferentes faixas raciais predominantes.

H₁: pelo menos uma média difere.
"""

# --- Carregar dados ---
df_final = pd.read_csv('tabelas/enade_2023_computacao_agregado.csv')

# Colunas das faixas de raça
faixas_renda = ['QE_I02_A','QE_I02_B','QE_I02_C','QE_I02_D','QE_I02_E','QE_I02_F']

# Criar uma coluna com a faixa de raça predominante em cada curso
df_final['faixa_raca_predom'] = df_final[faixas_renda].idxmax(axis=1)

# --- ANOVA ---
grupos = [df_final[df_final['faixa_raca_predom']==cat]['MEDIA_NT_CE']
          for cat in df_final['faixa_raca_predom'].unique()]

f_stat, p_val = stats.f_oneway(*grupos)
print("ANOVA:", f_stat, p_val)

# --- Teste Tukey ---
# O Tukey precisa dos dados em formato longo: uma coluna de notas e uma de grupos
tukey = pairwise_tukeyhsd(endog=df_final['MEDIA_NT_CE'],
                          groups=df_final['faixa_raca_predom'],
                          alpha=0.05)

print("\nResultado do teste de Tukey:")
print(tukey.summary())

# Preparar os dados para o MultiComparison
mc = MultiComparison(df_final['MEDIA_NT_CE'], df_final['faixa_raca_predom'])
tukey_result = mc.tukeyhsd()

# Gerar o gráfico
fig, ax = plt.subplots(figsize=(8, 6))
tukey_result.plot_simultaneous(ax=ax)
plt.title("Intervalos de Confiança - Teste de Tukey")
plt.xlabel("Diferença de Média")
plt.grid(True)
plt.show()

"""
ANOVA: 1.1347369708804156 0.3244043198769333

Resultado do teste de Tukey:
   Multiple Comparison of Means - Tukey HSD, FWER=0.05   
=========================================================
 group1   group2  meandiff p-adj   lower    upper  reject
---------------------------------------------------------
QE_I02_A QE_I02_B  -7.6174 0.6789 -29.0968 13.8621  False
QE_I02_A QE_I02_D   3.3148 0.4538  -3.2157  9.8453  False
QE_I02_B QE_I02_D  10.9321 0.4721 -11.1701 33.0343  False
---------------------------------------------------------

p = 0.324 (> 0.05) → não há evidência estatisticamente significativa de que a variável QE_I02 (raça/cor) esteja associada a diferenças nas notas CE, considerando os grupos analisados.

Ou seja, as médias entre os grupos de QE_I02 parecem semelhantes.
"""