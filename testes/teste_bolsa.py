import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt


"""
H₀: MEDIA_NT_CE não difere entre cursos com alunos que receberam bolsa de estudos.

H₁: pelo menos uma média difere.
"""

# --- Carregar dados ---
df_final = pd.read_csv('tabelas/enade_2023_computacao_agregado.csv')

# Colunas das faixas de raça
faixas_renda = ['QE_I13_A','QE_I13_B','QE_I13_C','QE_I13_D','QE_I13_E','QE_I13_F']

# Criar uma coluna com a faixa de raça predominante em cada curso
df_final['faixa_bolsa_predom'] = df_final[faixas_renda].idxmax(axis=1)

# --- ANOVA ---
grupos = [df_final[df_final['faixa_bolsa_predom']==cat]['MEDIA_NT_CE']
          for cat in df_final['faixa_bolsa_predom'].unique()]

f_stat, p_val = stats.f_oneway(*grupos)
print("ANOVA:", f_stat, p_val)

# --- Teste Tukey ---
# O Tukey precisa dos dados em formato longo: uma coluna de notas e uma de grupos
tukey = pairwise_tukeyhsd(endog=df_final['MEDIA_NT_CE'],
                          groups=df_final['faixa_bolsa_predom'],
                          alpha=0.05)

print("\nResultado do teste de Tukey:")
print(tukey.summary())

# Preparar os dados para o MultiComparison
mc = MultiComparison(df_final['MEDIA_NT_CE'], df_final['faixa_bolsa_predom'])
tukey_result = mc.tukeyhsd()

# Gerar o gráfico
fig, ax = plt.subplots(figsize=(8, 6))
tukey_result.plot_simultaneous(ax=ax)
plt.title("Intervalos de Confiança - Teste de Tukey")
plt.xlabel("Diferença de Média")
plt.grid(True)
plt.show()
