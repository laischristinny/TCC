# Análise de Desempenho ENADE 2023 - Cursos de Engenharia

> Este projeto realiza o pré-processamento, análise estatística e modelagem preditiva dos dados do ENADE 2023 para cursos de Engenharia. O objetivo é investigar fatores socioeconômicos, raciais e acadêmicos associados ao desempenho dos cursos, utilizando técnicas de agregação, visualização e machine learning.

## 📂 Estrutura de Pastas

```
.
├── dados/              # Arquivos originais do ENADE 2023 (microdados)
├── graficos/           # Scripts para geração de gráficos exploratórios
├── imagens/            # Imagens dos gráficos gerados
├── tabelas/            # Tabelas agregadas e intermediárias geradas
├── testes/             # Scripts de testes estatísticos (ex: ANOVA, Tukey)
│
├── main.py                     # Script principal de pré-processamento e agregação
├── modelo_xgboost.py           # Treinamento e avaliação de modelo XGBoost
├── random_forest.py            # Treinamento e avaliação de modelo Random Forest
├── regressao_linear_lasso.py   # Modelagem com regressão linear Lasso
├── regressao_linear_simples.py # Modelagem com regressão linear simples
└── rf_validacao_cruzada.py     # Random Forest com validação cruzada
```

## 🚀 Como Executar

### 1\. Pré-processamento dos Dados

Execute `main.py` para gerar a tabela agregada principal (`enade_2023_engenharia_agregado.csv`) na pasta `tabelas/`.

```bash
python main.py
```

### 2\. Modelagem Preditiva

Execute os scripts de modelagem para treinar e avaliar os modelos.

```bash
python modelo_xgboost.py
python random_forest.py
# etc.
```

### 3\. Visualização

Execute os scripts na pasta `graficos/` para gerar as visualizações exploratórias, que serão salvas em `imagens/`.

### 4\. Testes Estatísticos

Execute os scripts na pasta `testes/` para realizar análises como ANOVA e Tukey.

## 📋 Requisitos

  * Python 3.8+
  * Bibliotecas: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, statsmodels

Instale as dependências com:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn statsmodels
```

## 📊 Dados

Os arquivos de dados originais do ENADE devem ser colocados na pasta `dados/`. O script principal irá gerar as tabelas agregadas e salvá-las na pasta `tabelas/`.

## 📈 Resultados

  * Tabelas agregadas por curso.
  * Gráficos de correlação e distribuição de variáveis.
  * Modelos preditivos treinados e suas métricas de desempenho.
  * Resultados dos testes estatísticos de diferenças entre grupos.

## ✍️ Autoria

Projeto desenvolvido para a análise de dados do ENADE 2023 em cursos da área de Computação.

## ⚖️ Licença

Este projeto destina-se a uso acadêmico. Consulte as diretrizes do ENADE/INEP para restrições de uso e distribuição dos dados originais.
