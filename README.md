# 🎗️ Classificação de Câncer de Mama com Machine Learning (Supervisionado e Não Supervisionado)

Este projeto apresenta um pipeline completo de Ciência de Dados e Aprendizado de Máquina aplicado ao conjunto de dados **Breast Cancer Wisconsin (Diagnostic)** do `scikit-learn`. A abordagem combina técnicas de **Aprendizado Não Supervisionado (K-Means, Clusterização Hierárquica e PCA)** para engenharia de atributos e visualização, seguidas por **Modelos Supervisionados** com validação cruzada rigorosa, otimização de hiperparâmetros, ajuste de limiar clínico e análise de interpretabilidade para diagnóstico preditivo (**Benigno vs. Maligno**). 🧬💻

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaineCalabrezi13/Classificacao-Cancer-Mama-ML/blob/main/Modelos_Unidos.ipynb)

---

## 📋 Índice
1. [Configuração do Ambiente e Bibliotecas](#1-configuração-do-ambiente-e-bibliotecas)
2. [Carga e Estruturação do Dataset](#2-carga-e-estruturação-do-dataset)
3. [Análise Exploratória de Dados (EDA)](#3-análise-exploratória-de-dados-eda)
4. [Pré-processamento e Divisão dos Dados](#4-pré-processamento-e-divisão-dos-dados)
5. [Aprendizado Não Supervisionado e Engenharia de Atributos](#5-aprendizado-não-supervisionado-e-engenharia-de-atributos)
6. [Modelagem Supervisionada e Otimização (GridSearchCV)](#6-modelagem-supervisionada-e-otimização-gridsearchcv)
7. [Validação Cruzada Repetida e Testes Estatísticos](#7-validação-cruzada-repetida-e-testes-estatísticos)
8. [Ajuste de Limiar Clínico (Threshold Tuning)](#8-ajuste-de-limiar-clínico-threshold-tuning)
9. [Avaliação Final no Conjunto de Teste](#9-avaliação-final-no-conjunto-de-teste)
10. [Interpretabilidade, Análise de Erros e Exportação](#10-interpretabilidade-análise-de-erros-e-exportação)

---

## 🛠️ 1. Configuração do Ambiente e Bibliotecas

Nesta etapa, são importadas todas as dependências e registradas as versões do ambiente para reprodutibilidade:
* **Ambiente:** Python `3.13.18` | NumPy `2.4.6` | Pandas `2.2.3` | Scikit-learn `1.6.1` | SciPy `1.16.3`
* 📊 **Manipulação e Visualização:** `numpy`, `pandas`, `matplotlib`, `seaborn`
* ⚙️ **Pré-processamento e Pipelines:** `train_test_split`, `GridSearchCV`, `StratifiedKFold`, `RepeatedStratifiedKFold`, `cross_validate`, `StandardScaler`, `Pipeline`, `BaseEstimator`, `TransformerMixin`
* 🤖 **Modelos Supervisionados:** `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`
* 🧩 **Não Supervisionado & Redução de Dimensionalidade:** `KMeans`, `PCA`, `linkage`, `fcluster`
* 📏 **Métricas, Testes e Exportação:** `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `classification_report`, `silhouette_score`, `adjusted_rand_score`, `normalized_mutual_info_score`, `permutation_importance`, `scipy.stats`, `joblib`

---

## 📂 2. Carga e Estruturação do Dataset

O dataset original possui **569 amostras** e **30 atributos numéricos** extraídos de imagens digitalizadas de punção aspirativa por agulha fina (PAAF) de massas mamárias. 🔬

Para alinhar a modelagem à interpretação clínica (onde detectar malignidade é o evento crítico), a variável-alvo (`target`) original do `scikit-learn` foi invertida e remapeada:
* `0` $\rightarrow$ **Benigno** 🟢
* `1` $\rightarrow$ **Maligno (Classe Positiva)** 🔴

Após a criação das colunas `diagnosis_numeric` e `diagnosis_label`, o DataFrame estruturado conta com **569 linhas e 33 colunas**.

---

## 🔍 3. Análise Exploratória de Dados (EDA)

A inspeção detalhada revelou as seguintes características:

* 🧾 **Tipos de Dados:** 30 variáveis `float64`, 2 variáveis `int64` e 1 variável `object` (rótulo textual).
* ✅ **Integridade:** **0 valores ausentes** e **0 registros duplicados** em todo o conjunto de dados.
* ⚖️ **Balanceamento das Classes:**
  | Diagnóstico | Contagem | Percentual (%) |
  | :--- | :---: | :---: |
  | 🟢 **Benigno (0)** | 357 | 62,74% |
  | 🔴 **Maligno (1)** | 212 | 37,26% |
* 📊 **Análise de Outliers (Método IQR):** Foram analisados todos os 30 atributos; **29 apresentam outliers**, com destaque para `area error` (65 outliers / 11,42%), `radius error` (38 / 6,68%) e `perimeter error` (38 / 6,68%), enquanto apenas `worst concave points` não apresentou valores discrepantes. Os outliers foram mantidos por representarem variações morfológicas reais de tumores agressivos.

> 💡 **Insight Visual:** Os gráficos de distribuição e *boxplots* da variável `mean radius` confirmam que tumores malignos possuem valores medianos substancialmente maiores e maior dispersão. 📈

---

## ✂️ 4. Pré-processamento e Divisão dos Dados

Para garantir uma avaliação rigorosa e evitar vazamento de dados (*Data Leakage*) 🛡️, o conjunto foi dividido de forma estratificada (`stratify=y`, `random_state=42`) em três subconjuntos independentes:

1. 🏋️ **Treino (60%):** 341 amostras $\times$ 30 atributos
2. 🧪 **Validação (20%):** 114 amostras $\times$ 30 atributos
3. 🎯 **Teste (20%):** 114 amostras $\times$ 30 atributos

---

## 🧠 5. Aprendizado Não Supervisionado e Engenharia de Atributos

### 📐 5.1. Padronização e Escolha do Número de Clusters ($K$)
Os dados foram normalizados com `StandardScaler` ajustado **exclusivamente** no conjunto de treino. O algoritmo **K-Means** foi avaliado para $K \in \{2, 3, 4, 5, 6\}$ utilizando **Inércia** (Método do Cotovelo) e **Silhouette Score**:

| Valor de $K$ | Silhouette Score | Inércia |
| :---: | :---: | :---: |
| ⭐ **K = 2** | **0.3462** | **6856.86** |
| K = 3 | 0.3137 | 5963.06 |
| K = 4 | 0.2880 | 5463.11 |
| K = 5 | 0.1593 | 5034.81 |
| K = 6 | 0.1638 | 4684.62 |

* ✨ **Resultado:** O valor ótimo selecionado foi **$K = 2$** (*Silhouette Score* = `0.3462`), refletindo naturalmente a separação binária entre tumores benignos e malignos. O alinhamento externo com os rótulos reais também é verificado via **Adjusted Rand Index (ARI)** e **Normalized Mutual Information (NMI)**.

### 🚀 5.2. Feature Engineering Baseada em Distâncias (Sem Data Leakage)
Para evitar vazamento durante a validação cruzada, foi implementado um transformador customizado (`KMeansDistanceTransformer`) compatível com o `Pipeline` do `scikit-learn`. Ele ajusta o `StandardScaler` e o `KMeans` dentro de cada dobra (*fold*) e adiciona as distâncias euclidianas aos centroides (`dist_to_cluster_0` e `dist_to_cluster_1`), expandindo o espaço de atributos de **30 para 32 features**.

### 🎨 5.3. Visualização com PCA (2D) e Análise de Redundância
* **PCA (2D):** Projeção dos dados padronizados em duas componentes principais (`PCA_1` e `PCA_2`) para comparar visualmente os agrupamentos do K-Means com os diagnósticos clínicos reais. 🗺️
* **Clusterização Hierárquica de Atributos:** Análise de correlação de Spearman com agrupamento hierárquico (*Ward*) para identificar blocos de colinearidade entre as 30 variáveis originais.

---

## 🤖 6. Modelagem Supervisionada e Otimização (GridSearchCV)

Quatro famílias de classificadores foram avaliadas sob **dois cenários** — **Originais (30 features)** vs. **Com K-Means (32 features)** — utilizando `StratifiedKFold(n_splits=5)` e otimização via `GridSearchCV` (com foco em `f1` / `roc_auc`):

* 🌳 **Decision Tree (`DecisionTreeClassifier`)**
* 🌲 **Random Forest (`RandomForestClassifier`)**
* 🚀 **Gradient Boosting (`GradientBoostingClassifier`)**
* 📐 **Support Vector Machine (`SVC` com `probability=True`)**

Todos os modelos foram encapsulados em objetos `Pipeline` garantindo que escalonamento e engenharia de atributos ocorram estritamente dentro de cada partição de treino.

---

## 📊 7. Validação Cruzada Repetida e Testes Estatísticos

Para verificar se a adição das *features* de distância do K-Means gera ganho real ou apenas variação aleatória, os melhores estimadores de cada cenário foram submetidos a uma **Validação Cruzada Estratificada Repetida (`RepeatedStratifiedKFold`, 5 folds $\times$ 10 repetições = 50 avaliações)** nos dados de treino + validação:

* 📏 **Métricas Avaliadas:** `Accuracy`, `Precision`, `Recall`, `F1-Score` e `ROC-AUC`.
* 🔬 **Testes Pareados:** Aplicação do **Teste t pareado** e do **Teste de Wilcoxon** comparando o desempenho de cada modelo com e sem as *features* do K-Means, verificando a significância estatística ($p < 0.05$).

---

## 🎚️ 8. Ajuste de Limiar Clínico (Threshold Tuning)

Em triagem oncológica, **falsos negativos** (classificar um tumor maligno como benigno) representam o maior risco ao paciente. Por isso, no conjunto de **Validação**, foi realizada uma varredura de limiares de probabilidade ($\theta \in [0.10, 0.90]$) para selecionar o ponto de corte que maximiza o **F1-Score** ou atinge uma meta mínima de **Sensibilidade (Recall $\ge 0.95$)**, comparando o comportamento em relação ao limiar padrão (`0.50`).

---

## 🎯 9. Avaliação Final no Conjunto de Teste

O modelo campeão selecionado na etapa de validação é avaliado uma única vez no conjunto de **Teste (114 amostras inéditas)**, gerando:
* Relatório completo de classificação (`Precision`, `Recall`, `F1-Score`, `Accuracy` e `ROC-AUC`).
* **Matriz de Confusão** comparando o limiar padrão (`0.50`) e o limiar clínico ajustado.
* **Curva ROC** e **Curva Precision-Recall**.

---

## 🔎 10. Interpretabilidade, Análise de Erros e Exportação

* 🧠 **Importância de Atributos (`Permutation Importance`):** Identificação das variáveis morfológicas que mais impactam a decisão do modelo campeão no conjunto de teste.
* 🩺 **Análise de Falsos Negativos e Falsos Positivos:** Inspeção individual dos casos classificados incorretamente, exibindo suas probabilidades preditas e valores clínicos para entender os casos limítrofes.
* 💾 **Persistência do Modelo:** O pipeline final treinado e seus metadados (limiar ótimo, métricas e lista de atributos) são salvos com `joblib` (`modelo_campeao_cancer_mama.joblib`) para uso em produção ou inferência futura.
