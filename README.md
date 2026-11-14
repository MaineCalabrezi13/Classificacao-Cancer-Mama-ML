# 🎗️ Classificação de Câncer de Mama com Machine Learning (Supervisionado e Não Supervisionado)

Este projeto apresenta um pipeline completo de Ciência de Dados e Aprendizado de Máquina aplicado ao conjunto de dados **Breast Cancer Wisconsin (Diagnostic)** do `scikit-learn`. A abordagem combina técnicas de **Aprendizado Não Supervisionado (K-Means e PCA)** para engenharia de atributos e visualização, seguidas por **Modelos Supervisionados** para diagnóstico preditivo (Benigno vs. Maligno). 🧬💻

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaineCalabrezi13/Classificacao-Cancer-Mama-ML/blob/main/Modelos_Unidos.ipynb)

---

## 📋 Índice
1. [Configuração do Ambiente e Bibliotecas](#1-configuração-do-ambiente-e-bibliotecas)
2. [Carga e Estruturação do Dataset](#2-carga-e-estruturação-do-dataset)
3. [Análise Exploratória de Dados (EDA)](#3-análise-exploratória-de-dados-eda)
4. [Pré-processamento e Divisão dos Dados](#4-pré-processamento-e-divisão-dos-dados)
5. [Aprendizado Não Supervisionado e Engenharia de Atributos](#5-aprendizado-não-supervisionado-e-engenharia-de-atributos)

---

## 🛠️ 1. Configuração do Ambiente e Bibliotecas

Nesta etapa, são importadas todas as dependências necessárias para manipulação de dados, visualização gráfica, pré-processamento, clusterização, redução de dimensionalidade e classificação supervisionada.

* 📊 **Manipulação e Visualização:** `numpy`, `pandas`, `matplotlib`, `seaborn`
* ⚙️ **Pré-processamento e Validação:** `train_test_split`, `GridSearchCV`, `StratifiedKFold`, `cross_val_score`, `StandardScaler`, `Pipeline`
* 🤖 **Modelos Supervisionados:** `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`
* 🧩 **Modelos Não Supervisionados & Dimensionalidade:** `KMeans`, `PCA`
* 📏 **Métricas de Avaliação:** `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `roc_curve`, `silhouette_score`

---

## 📂 2. Carga e Estruturação do Dataset

O dataset original possui **569 amostras** e **30 atributos numéricos** extraídos de imagens digitalizadas de punção aspirativa por agulha fina (PAAF) de massas mamárias. 🔬

Para facilitar a interpretação clínica, a variável-alvo (`target`) foi remapeada:
* `0` $\rightarrow$ **Benigno** 🟢
* `1` $\rightarrow$ **Maligno** 🔴

Após a criação das colunas `diagnosis_numeric` e `diagnosis_label`, o DataFrame final conta com **569 linhas e 33 colunas**.

---

## 🔍 3. Análise Exploratória de Dados (EDA)

A inspeção inicial revelou as seguintes características estruturais e estatísticas:

* 🧾 **Tipos de Dados:** 30 variáveis `float64`, 2 variáveis `int64` e 1 variável `object` (rótulo textual).
* ✅ **Valores Ausentes:** **0 valores nulos** em todo o conjunto de dados (não há necessidade de imputação).
* ⚖️ **Balanceamento das Classes:**
  | Diagnóstico | Contagem | Percentual (%) |
  | :--- | :---: | :---: |
  | 🟢 **Benigno (0)** | 357 | 62,74% |
  | 🔴 **Maligno (1)** | 212 | 37,26% |

> 💡 **Insight Visual:** A análise exploratória também contempla gráficos de contagem das classes e *boxplots* da variável `mean radius`, evidenciando que tumores malignos apresentam valores significativamente maiores e maior dispersão/presença de *outliers*. 📈

---

## ✂️ 4. Pré-processamento e Divisão dos Dados

Para garantir uma avaliação rigorosa e evitar vazamento de dados (*Data Leakage*) 🛡️, o conjunto foi dividido de forma estratificada (`stratify=y`, `random_state=42`) em três subconjuntos:

1. 🏋️ **Treino (60%):** 341 amostras $\times$ 30 atributos
2. 🧪 **Validação (20%):** 114 amostras $\times$ 30 atributos
3. 🎯 **Teste (20%):** 114 amostras $\times$ 30 atributos

---

## 🧠 5. Aprendizado Não Supervisionado e Engenharia de Atributos

### 📐 5.1. Padronização e Escolha do Número de Clusters ($K$)
Os dados foram normalizados com `StandardScaler` ajustado **exclusivamente** no conjunto de treino. Em seguida, o algoritmo **K-Means** foi avaliado para $K \in \{2, 3, 4, 5, 6\}$ utilizando as métricas de **Inércia** (Método do Cotovelo) e **Silhouette Score**:

| Valor de $K$ | Silhouette Score | Inércia |
| :---: | :---: | :---: |
| ⭐ **K = 2** | **0
