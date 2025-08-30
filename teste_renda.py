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

# --- Carregar dados ---
df_final = pd.read_csv('enade_2023_computacao_agregado.csv')

# Colunas das faixas de renda
faixas_renda = ['QE_I08_A','QE_I08_B','QE_I08_C','QE_I08_D','QE_I08_E','QE_I08_F','QE_I08_G']

# Criar uma coluna com a faixa de renda predominante em cada curso
df_final['faixa_renda_predom'] = df_final[faixas_renda].idxmax(axis=1)

# --- ANOVA ---
grupos = [df_final[df_final['faixa_renda_predom']==cat]['MEDIA_NT_CE']
          for cat in df_final['faixa_renda_predom'].unique()]

f_stat, p_val = stats.f_oneway(*grupos)
print("ANOVA:", f_stat, p_val)

# --- Teste Tukey ---
# O Tukey precisa dos dados em formato longo: uma coluna de notas e uma de grupos
tukey = pairwise_tukeyhsd(endog=df_final['MEDIA_NT_CE'],
                          groups=df_final['faixa_renda_predom'],
                          alpha=0.05)

print("\nResultado do teste de Tukey:")
print(tukey.summary())

# Preparar os dados para o MultiComparison
mc = MultiComparison(df_final['MEDIA_NT_CE'], df_final['faixa_renda_predom'])
tukey_result = mc.tukeyhsd()

# Gerar o gráfico
fig, ax = plt.subplots(figsize=(8, 6))
tukey_result.plot_simultaneous(ax=ax)
plt.title("Intervalos de Confiança - Teste de Tukey")
plt.xlabel("Diferença de Média")
plt.grid(True)
plt.show()

""" Resultado: ANOVA: 454.0696239197707 0.0 454.06 → é o F-statistic (estatística de teste). 
Ele mede a razão entre a variabilidade entre os grupos (diferença de médias entre faixas de renda) e a variabilidade dentro dos grupos (diferença entre cursos dentro da mesma faixa de renda). 
Quanto maior esse valor, mais provável que exista uma diferença real entre as médias. 
O teste ANOVA foi significativo, F(…)=454.07, p < 0.001. 
Portanto, existe diferença estatisticamente significativa entre as médias de desempenho de cursos com diferentes perfis de renda. 
"""

"""
Tukey:
a) QE_I08_A (faixa de renda mais baixa) vs outras:

A vs B (3.54, p<0.001, True): cursos com predominância da faixa A têm, em média, notas 3,5 pontos menores que cursos com predominância da faixa B → diferença significativa.

A vs C (0.93, p=0.803, False): praticamente não há diferença significativa entre A e C.

A vs E (7.08, p<0.001, True): diferença grande → cursos com renda E (mais alta) têm notas ~7 pontos maiores que cursos de renda A.

A vs F (18.06, p<0.001, True): cursos da faixa F (renda mais alta ainda) têm notas ~18 pontos maiores que cursos de renda A.

📌 Isso mostra que quanto maior a renda predominante, maior a nota média dos cursos.
"""