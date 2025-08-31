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

""" Hipótese nula (H₀): não há diferença de desempenho médio entre cursos de computação com diferentes composições raciais predominantes.

Resultado: p ≪ 0,05 → rejeitamos H₀.

✅ Conclusão: há diferenças significativas no desempenho médio (MEDIA_NT_CE) entre cursos cuja maioria dos alunos se autodeclara de diferentes grupos raciais.
"""

"""
A vs B (meandiff = -8.88, p=0.0102, reject=True):
Cursos em que predomina o grupo A têm desempenho em média 8,9 pontos menor que os cursos em que predomina o grupo B → diferença significativa.

A vs D (meandiff = 3.10, p<0.001, reject=True):
Cursos com predominância do grupo D têm notas ~3 pontos maiores que os do grupo A → diferença significativa.

B vs D (meandiff = 11.99, p<0.001, reject=True):
Cursos do grupo D têm notas em média ~12 pontos maiores que os do grupo B → diferença significativa.
"""