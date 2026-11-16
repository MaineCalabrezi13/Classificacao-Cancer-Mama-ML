# 🎗️ Classificação de Câncer de Mama com Machine Learning (Supervisionado e Não Supervisionado)

Este projeto apresenta um estudo de modelos de Machine Learning aplicados ao conjunto de dados **Breast Cancer Wisconsin (Diagnostic) (WDBC)**, disponibilizado pela biblioteca `scikit-learn`. O repositório contém dois notebooks complementares:

- **`Modelos_Unidos.ipynb`:** compara quatro classificadores supervisionados — Árvore de Decisão, Support Vector Machine (SVM), Random Forest e Gradient Boosting — nos cenários **Sem K-Means** e **Com K-Means**. Também utiliza K-Means em análises exploratórias de agrupamento e PCA para visualização bidimensional.
- **`Modelo_Complementar.ipynb`:** realiza um experimento adicional de calibração do limiar de decisão por meio de previsões *Out-of-Fold* (OOF), avaliando diferentes metas de sensibilidade (Recall) e comparando os oito pipelines. Nesse experimento, o modelo selecionado foi o SVM com K-Means.

Em conjunto, os experimentos investigam tanto o efeito dos atributos derivados das distâncias aos centroides do K-Means quanto a relação entre limiar de decisão e identificação de tumores malignos. Os modelos são ferramentas experimentais de apoio à análise de dados e **não substituem avaliação, diagnóstico ou decisão de profissionais de saúde**.

[![Open In Colab - Modelos Unidos](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaineCalabrezi13/Classificacao-Cancer-Mama-ML/blob/main/Modelos_Unidos.ipynb)
[![Open In Colab - Modelo Complementar](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaineCalabrezi13/Classificacao-Cancer-Mama-ML/blob/main/Modelo_Complementar.ipynb)

---

## 📋 Índice

1. [Configuração do ambiente e bibliotecas](#-1-configuração-do-ambiente-e-bibliotecas)
2. [Carga e estruturação do dataset](#-2-carga-e-estruturação-do-dataset)
3. [Análise exploratória de dados (EDA)](#-3-análise-exploratória-de-dados-eda)
4. [Pré-processamento e divisão dos dados](#-4-pré-processamento-e-divisão-dos-dados)
5. [Aprendizado não supervisionado e engenharia de atributos](#-5-aprendizado-não-supervisionado-e-engenharia-de-atributos)
6. [Modelagem supervisionada e otimização](#-6-modelagem-supervisionada-e-otimização)
7. [Avaliação dos modelos](#-7-avaliação-dos-modelos)
8. [Validação cruzada aninhada e comparação estatística](#-8-validação-cruzada-aninhada-e-comparação-estatística)
9. [Importância dos atributos](#-9-importância-dos-atributos)
10. [Experimento complementar: ajuste do limiar de decisão](#-10-experimento-complementar-ajuste-do-limiar-de-decisão)
11. [Como executar](#-11-como-executar)
12. [Conclusões](#-12-conclusões)

---

## 🛠️ 1. Configuração do ambiente e bibliotecas

O notebook foi desenvolvido em Python e utiliza as seguintes bibliotecas:

- **Manipulação de dados:** `numpy` e `pandas`.
- **Visualização:** `matplotlib` e `seaborn`.
- **Pré-processamento e pipelines:** `StandardScaler`, `Pipeline`, `BaseEstimator`, `TransformerMixin` e `clone`.
- **Validação e otimização:** `train_test_split`, `GridSearchCV`, `StratifiedKFold` e `RepeatedStratifiedKFold`.
- **Modelos supervisionados:** `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier` e `SVC`.
- **Modelos não supervisionados:** `KMeans` e `PCA`.
- **Métricas:** acurácia, precisão, recall, F1-Score, ROC-AUC, matriz de confusão, Adjusted Rand Index (ARI), Normalized Mutual Information (NMI) e Silhouette Score.
- **Análises complementares:** `permutation_importance`, testes estatísticos do `scipy`, agrupamento hierárquico e persistência com `joblib`.

Versões registradas na execução do notebook: Python 3.13.18, NumPy 2.4.6, Pandas 2.2.3, scikit-learn 1.6.1 e SciPy 1.16.3. Outras versões podem funcionar, mas podem produzir pequenas diferenças numéricas ou de compatibilidade.

---

## 📂 2. Carga e estruturação do dataset

O conjunto é carregado diretamente pela função `load_breast_cancer(as_frame=True)`, do `scikit-learn`, sem necessidade de baixar um arquivo externo.

O dataset contém:

- **569 amostras**;
- **30 atributos numéricos preditores**;
- a coluna original `target`;
- duas colunas derivadas pelo notebook: `diagnosis_numeric` e `diagnosis_label`.

Assim, o DataFrame estruturado possui **569 linhas e 33 colunas**.

### Codificação do diagnóstico

É importante distinguir a codificação original do `scikit-learn` da variável criada no projeto:

| Variável | Valor | Diagnóstico |
|---|---:|---|
| `target` (original) | 0 | Maligno |
| `target` (original) | 1 | Benigno |
| `diagnosis_numeric` (remapeada) | 0 | Benigno |
| `diagnosis_numeric` (remapeada) | 1 | Maligno |

O remapeamento realizado no notebook é:

```python
df['diagnosis_numeric'] = df['target'].map({0: 1, 1: 0})
df['diagnosis_label'] = df['diagnosis_numeric'].map({
    1: 'Maligno',
    0: 'Benigno'
})
```

Portanto, **Maligno (1) é a classe positiva** utilizada nas métricas, nas matrizes de confusão e nas avaliações dos classificadores.

---

## 🔍 3. Análise exploratória de dados (EDA)

A inspeção inicial apresentou os seguintes resultados:

| Verificação | Resultado |
|---|---:|
| Total de amostras | 569 |
| Atributos preditores | 30 |
| Colunas no DataFrame estruturado | 33 |
| Valores ausentes | 0 |
| Linhas completamente duplicadas | 0 |
| Duplicatas considerando somente os 30 atributos | 0 |

### Distribuição das classes

| Diagnóstico | Contagem | Percentual |
|---|---:|---:|
| Benigno (0) | 357 | 62,74% |
| Maligno (1) | 212 | 37,26% |
| **Total** | **569** | **100,00%** |

### Análise de valores extremos

Foi utilizado o método do Intervalo Interquartil (IQR), com limites definidos por `Q1 - 1,5 × IQR` e `Q3 + 1,5 × IQR`, aplicado aos 30 atributos preditores. Foram identificados pontos classificados como outliers em **29 dos 30 atributos**. O atributo `area error` apresentou a maior quantidade, com **65 ocorrências (11,42%)**. O atributo `worst concave points` não apresentou outliers segundo esse critério.

Os pontos identificados foram mantidos no conjunto de dados; não houve remoção automática de outliers. A EDA também inclui gráficos de distribuição das classes, boxplot de `mean radius` por diagnóstico e gráfico da quantidade de outliers por atributo.

---

## ✂️ 4. Pré-processamento e divisão dos dados

Foram utilizados os 30 atributos originais como entrada (`X`) e `diagnosis_numeric` como variável-alvo (`y`). A divisão foi estratificada, com `random_state=42`, em três subconjuntos:

| Subconjunto | Proporção do total | Amostras |
|---|---:|---:|
| Treino | 60% | 341 |
| Validação | 20% | 114 |
| Teste | 20% | 114 |
| **Total** | **100%** | **569** |

O teste foi separado inicialmente e reservado para a avaliação final. O conjunto de validação foi utilizado na comparação dos modelos e na seleção do modelo final. O conjunto de treino foi utilizado para ajuste dos transformadores, busca de hiperparâmetros e validação cruzada aninhada.

A padronização com `StandardScaler` é realizada dentro dos pipelines, de modo que o scaler seja ajustado somente nos dados de treinamento de cada ajuste/fold e aplicado posteriormente às respectivas partições de validação ou teste. Essa organização reduz o risco de *data leakage*.

---

## 🧠 5. Aprendizado não supervisionado e engenharia de atributos

O K-Means é utilizado de duas maneiras distintas: (1) análise exploratória de agrupamentos e (2) transformador de engenharia de atributos dentro dos pipelines preditivos. Os dois procedimentos não devem ser confundidos.

### 5.1. Análise exploratória da escolha de K

Na análise exploratória, os 30 atributos do conjunto de treino são padronizados e o K-Means é avaliado para `K ∈ {2, 3, 4, 5, 6}`, utilizando Inércia e Silhouette Score.

| K | Silhouette Score | Inércia |
|---:|---:|---:|
| **2** | **0.3462** | **6856.86** |
| 3 | 0.3137 | 5963.06 |
| 4 | 0.2880 | 5463.11 |
| 5 | 0.1593 | 5034.81 |
| 6 | 0.1638 | 4684.62 |

Na exploração, **K = 2** apresentou o maior Silhouette Score entre os valores testados. Esse resultado é descritivo e foi calculado somente no conjunto de treino.

### 5.2. K-Means dentro dos pipelines

Na modelagem, o valor de K **não é fixado em 2** e não é simplesmente reutilizado da análise exploratória. O transformador customizado `ClusterDistanceFeatures` seleciona K internamente, avaliando os valores de 2 a 6 pelo Silhouette Score durante seu método `fit()`.

Essa seleção ocorre somente com os dados disponíveis no treinamento de cada fold. Assim, as amostras do fold de validação não participam da escolha de K nem do ajuste do K-Means.

O transformador acrescenta ao conjunto de atributos as distâncias euclidianas de cada amostra aos centroides encontrados. Portanto, a quantidade de atributos derivados depende do K selecionado:

| K selecionado | Atributos originais | Distâncias adicionadas | Total |
|---:|---:|---:|---:|
| 2 | 30 | 2 | 32 |
| 3 | 30 | 3 | 33 |
| 4 | 30 | 4 | 34 |
| 5 | 30 | 5 | 35 |
| 6 | 30 | 6 | 36 |

Após a geração das distâncias, o transformador `ScaleClusterDistances` padroniza essas colunas derivadas. O pipeline, portanto, mantém as 30 variáveis originais e acrescenta as distâncias aos centroides selecionados naquele ajuste.

Na execução registrada da Nested CV, os valores selecionados nos folds externos foram **K = 2 em 60 ajustes**. Esse resultado observado não altera o fato de que a implementação permite a seleção de K entre 2 e 6 em cada ajuste.

### 5.3. Concordância dos agrupamentos com os diagnósticos

Na análise exploratória com K = 2, a concordância entre os agrupamentos e os diagnósticos reais foi avaliada por:

- **Adjusted Rand Index (ARI): 0.6473**;
- **Normalized Mutual Information (NMI): 0.5246**.

Essas métricas indicam concordância moderada a boa, mas não significam que os clusters reproduzam perfeitamente os diagnósticos. O K-Means não utiliza os rótulos na formação dos grupos.

### 5.4. Visualização com PCA

O PCA é utilizado para projetar os 30 atributos padronizados em duas dimensões (`PCA_1` e `PCA_2`) e permitir a inspeção visual dos agrupamentos e dos diagnósticos. A projeção é uma ferramenta de visualização; não constitui, por si só, uma métrica de desempenho dos classificadores.

---

## 🤖 6. Modelagem supervisionada e otimização

Foram avaliados quatro classificadores em dois cenários:

1. **Sem K-Means:** 30 atributos originais padronizados.
2. **Com K-Means:** 30 atributos originais, acrescidos das distâncias aos centroides selecionadas internamente pelo transformador.

Os oito pipelines são construídos de forma modular, integrando pré-processamento, engenharia de atributos condicional e classificador.

### 6.1. Modelos

- **Árvore de Decisão (`DecisionTreeClassifier`):** classificador baseado em divisões sucessivas dos atributos.
- **SVM (`SVC`):** classificador com kernel RBF e probabilidade habilitada, utilizado para modelar fronteiras de decisão não lineares.
- **Random Forest (`RandomForestClassifier`):** conjunto de árvores treinadas com aleatoriedade de amostras e atributos.
- **Gradient Boosting (`GradientBoostingClassifier`):** conjunto de árvores construídas sequencialmente, em que novos estimadores buscam corrigir erros anteriores.

### 6.2. Busca de hiperparâmetros

Foi utilizado `GridSearchCV` com validação cruzada estratificada de 5 folds (`StratifiedKFold`, `cv=5`) no conjunto de treino. O F1-Score foi a métrica de seleção (`scoring='f1'` e `refit=True`), com monitoramento adicional de Recall e Acurácia.

As grades avaliadas foram:

| Modelo | Hiperparâmetro | Valores testados |
|---|---|---|
| Árvore de Decisão | `max_depth` | `[3, 5, 7, None]` |
| Árvore de Decisão | `min_samples_leaf` | `[1, 2, 5]` |
| SVM (RBF) | `C` | `[0.1, 1, 10]` |
| SVM (RBF) | `gamma` | `['scale', 0.01, 0.1]` |
| Random Forest | `n_estimators` | `[100, 300]` |
| Random Forest | `max_depth` | `[None, 5, 10]` |
| Random Forest | `min_samples_leaf` | `[1, 2]` |
| Gradient Boosting | `n_estimators` | `[100, 200]` |
| Gradient Boosting | `learning_rate` | `[0.05, 0.1]` |
| Gradient Boosting | `max_depth` | `[2, 3]` |

### 6.3. Seleção do modelo final

Os modelos otimizados foram comparados no conjunto de validação segundo critérios hierárquicos: maior F1-Score, seguido de Recall e ROC-AUC. O modelo selecionado foi o **SVM sem K-Means**. Após a seleção, ele foi congelado e avaliado no conjunto de teste. Os demais modelos foram avaliados no teste somente para comparação descritiva.

O modelo final foi serializado com `joblib` no arquivo `modelo_final_cancer_mama.joblib`.

---

## 📊 7. Avaliação dos modelos

As métricas utilizadas foram:

- **Acurácia:** proporção total de classificações corretas.
- **Precisão:** proporção de previsões positivas que realmente são malignas.
- **Recall (sensibilidade):** proporção de tumores malignos identificados corretamente.
- **F1-Score:** média harmônica entre precisão e recall.
- **ROC-AUC:** capacidade de discriminação entre as classes em diferentes limiares.

### 7.1. Comparação no conjunto de validação

Os resultados registrados para a seleção do modelo foram:

| Modelo | Cenário | F1-Score | Recall |
|---|---|---:|---:|
| Árvore de Decisão | Sem K-Means | 0.8764 | 0.9070 |
| Árvore de Decisão | Com K-Means | 0.8837 | 0.8837 |
| Random Forest | Sem K-Means | 0.9647 | 0.9535 |
| Random Forest | Com K-Means | 0.9535 | 0.9535 |
| Gradient Boosting | Sem K-Means | 0.9213 | 0.9535 |
| Gradient Boosting | Com K-Means | 0.9213 | 0.9535 |
| SVM | Sem K-Means | **0.9655** | **0.9767** |
| SVM | Com K-Means | 0.9545 | 0.9767 |

O SVM sem K-Means apresentou o maior F1-Score na validação (0.9655) e foi selecionado como modelo final. O Recall foi de 0.9767 nos dois cenários do SVM, portanto houve empate nessa métrica. A seleção do cenário sem K-Means decorreu do F1-Score mais alto na validação, e não de uma superioridade em todas as métricas.

### 7.2. Comparação descritiva no conjunto de teste

| Modelo | Cenário | Acurácia |
|---|---|---:|
| Árvore de Decisão | Sem K-Means | 92.98% |
| Árvore de Decisão | Com K-Means | 92.11% |
| Random Forest | Sem K-Means | 95.61% |
| Random Forest | Com K-Means | 95.61% |
| Gradient Boosting | Sem K-Means | 96.49% |
| Gradient Boosting | Com K-Means | 96.49% |
| SVM | Sem K-Means | **98.25%** |
| SVM | Com K-Means | 98.25% |

Os resultados no teste são apresentados como comparação descritiva. Os dois cenários do SVM alcançaram a mesma acurácia (98.25%). Portanto, com base na acurácia do teste, não é possível afirmar que o SVM sem K-Means foi superior ao SVM com K-Means. A escolha do modelo final foi feita anteriormente, com base no conjunto de validação, em que o cenário sem K-Means obteve F1-Score maior (0.9655 contra 0.9545), enquanto o Recall foi igual (0.9767). A igualdade de acurácia não significa necessariamente que todas as demais métricas ou previsões dos dois cenários sejam idênticas.

### 7.3. Avaliação detalhada do modelo final

O SVM com kernel RBF, sem K-Means, obteve no conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Acurácia | 98.25% |
| Precisão | 100.00% |
| Recall | 95.24% |
| F1-Score | 97.56% |
| ROC-AUC | 0.9957 |

A matriz de confusão, considerando Maligno como classe positiva, foi:

| | Predito benigno | Predito maligno |
|---|---:|---:|
| **Real benigno** | 72 | 0 |
| **Real maligno** | 2 | 40 |

Assim, foram registrados 72 verdadeiros negativos, nenhum falso positivo, 2 falsos negativos e 40 verdadeiros positivos. Dois dos 42 casos malignos do conjunto de teste não foram identificados pelo modelo.

### 7.4. Intervalos de confiança bootstrap

Foram calculados intervalos de confiança de 95% por bootstrap estratificado, com 2.000 reamostragens:

| Métrica | Limite inferior | Limite superior |
|---|---:|---:|
| Acurácia | 0.9561 | 1.0000 |
| Precisão | 1.0000 | 1.0000 |
| Recall | 0.8810 | 1.0000 |
| F1-Score | 0.9367 | 1.0000 |
| ROC-AUC | 0.9851 | 1.0000 |

A amplitude do intervalo de Recall deve ser interpretada considerando o tamanho do conjunto de teste, que contém 42 casos malignos.

---

## 🧪 8. Validação cruzada aninhada e comparação estatística

Foi realizada Nested Cross-Validation para comparar, de forma pareada, os cenários Sem K-Means e Com K-Means. Essa etapa utiliza somente o conjunto de treino; o conjunto de validação e o teste final ficam fora da Nested CV.

Configuração:

- **Folds externos:** 5;
- **Repetições externas:** 3;
- **Total de divisões externas pareadas:** 15 (`J = 15`);
- **Folds internos:** 3;
- **Comparações:** 12 (4 modelos × 3 métricas: F1, Recall e ROC-AUC);
- **Teste estatístico:** teste t pareado corrigido de Nadeau & Bengio;
- **Correção para múltiplas comparações:** Holm-Bonferroni;
- **Nível de significância:** `α = 0.05`.

Em cada fold externo, a busca de hiperparâmetros ocorre somente no treinamento externo. Dentro de cada ajuste, o escalonamento, a escolha de K e o K-Means são ajustados sem utilizar os dados do fold externo de avaliação. Os mesmos folds são utilizados nos dois cenários para permitir a comparação pareada.

### Desempenho médio nos folds externos

Valores de F1-Score apresentados como média dos folds externos:

| Modelo | Sem K-Means | Com K-Means |
|---|---:|---:|
| Árvore de Decisão | 0.9117 ± 0.0470 | 0.9069 ± 0.0469 |
| Random Forest | 0.9300 ± 0.0312 | 0.9239 ± 0.0309 |
| Gradient Boosting | 0.9478 ± 0.0231 | 0.9420 ± 0.0340 |
| SVM | 0.9547 ± 0.0296 | 0.9601 ± 0.0273 |

Após o teste pareado corrigido e a correção de Holm-Bonferroni, **nenhuma das 12 comparações apresentou diferença estatisticamente significativa** entre Sem K-Means e Com K-Means.

A ausência de evidência estatística de diferença **não comprova equivalência** entre os cenários. O próprio procedimento ressalta que o número de folds externos correlacionados (J = 15) limita o poder estatístico.

---

## 🔬 9. Importância dos atributos

A importância dos atributos foi analisada por permutação no conjunto de validação, utilizando o modelo final. A análise individual utiliza 50 repetições e mede a queda do F1-Score após a permutação de cada atributo.

Como atributos morfométricos podem ser fortemente correlacionados — por exemplo, raio, perímetro e área — também foi realizada uma análise por grupos de atributos correlacionados. Os grupos foram definidos a partir da correlação absoluta de Spearman superior a 0,9, calculada no conjunto de treino. Na permutação por grupo, as colunas do grupo são permutadas conjuntamente.

Os grupos com maior queda média de F1-Score registrados foram:

| Grupo de atributos | Queda média de F1-Score |
|---|---:|
| Tamanho (mean/worst radius, perimeter e area) | 0.1497 ± 0.0350 |
| Concavidade | 0.0482 |
| Textura | 0.0441 |
| Erros de raio/perímetro/área | 0.0428 |
| Pontos côncavos | 0.0237 |
| Compacidade | -0.0034 |

A análise sugere maior contribuição das dimensões geométricas para a classificação. Como essa análise interpretativa foi realizada na validação, ela não deve ser tratada como uma estimativa independente de desempenho em dados externos.

---

## 🎯 10. Experimento complementar: ajuste do limiar de decisão

O notebook `Modelo_Complementar.ipynb` amplia a análise dos classificadores por meio da investigação do ajuste do limiar de decisão. O objetivo é avaliar como diferentes limiares influenciam a classificação das amostras como Benignas ou Malignas, especialmente em relação à sensibilidade (Recall).

Diferentemente do experimento principal, esta etapa realiza a calibração do limiar a partir de previsões **Out-of-Fold (OOF)** geradas exclusivamente com os dados de treinamento. Foram considerados alvos de Recall de 95%, 97,5%, 98% e 100%.

A seleção do modelo considerou os resultados da validação, utilizando critérios hierárquicos de F1-Score, Recall e ROC-AUC. O modelo selecionado foi o **SVM com K-Means**, com limiar de decisão de aproximadamente **0,2717**.

### 10.1. Calibração do limiar

Para a meta de Recall de 98%, o procedimento de calibração OOF selecionou o limiar de aproximadamente 0,2717 para o SVM com K-Means. Nas previsões OOF utilizadas nessa calibração, foram identificados 125 dos 127 casos malignos, correspondendo a Recall de 98,43%.

Esse resultado descreve o desempenho nas previsões OOF de calibração e não representa uma garantia de que a mesma meta será atingida em dados independentes.

### 10.2. Resultados no conjunto de teste

O SVM com K-Means, utilizando o limiar ajustado, apresentou:

| Métrica | Resultado |
|---|---:|
| Acurácia | 94,74% |
| Precisão | 89,13% |
| Recall | 97,62% |
| F1-Score | 93,18% |
| ROC-AUC | 0,9957 |
| Limiar de decisão | 0,2717 |

A matriz de confusão, considerando Maligno como classe positiva, foi:

| | Predito benigno | Predito maligno |
|---|---:|---:|
| **Real benigno** | 67 | 5 |
| **Real maligno** | 1 | 41 |

Assim, o modelo identificou 41 dos 42 casos malignos, registrando 1 falso negativo e 5 falsos positivos.

### 10.3. Comparação entre os limiares de decisão

Para avaliar o efeito do ajuste do limiar, foram comparados os resultados do mesmo modelo (SVM com K-Means) no conjunto de teste, utilizando o limiar padrão de 0,5 e o limiar calibrado de 0,2717.

| Métrica | Limiar 0,5 | Limiar 0,2717 |
|---|---:|---:|
| Acurácia | 98,25% | 94,74% |
| Precisão | 97,62% | 89,13% |
| Recall | 97,62% | 97,62% |
| F1-Score | 97,62% | 93,18% |
| ROC-AUC | 0,9957 | 0,9957 |
| Falsos negativos | 1 | 1 |
| Falsos positivos | 1 | 5 |

Neste conjunto de teste, a redução do limiar não aumentou o Recall nem reduziu os falsos negativos. Ambos os limiares identificaram 41 dos 42 casos malignos. Entretanto, o limiar ajustado aumentou os falsos positivos de 1 para 5, reduzindo a precisão, a acurácia e o F1-Score.

Esse resultado demonstra que a calibração realizada com previsões OOF, embora tenha atingido a meta de Recall nos dados de calibração, não garante que o mesmo ganho seja observado em um conjunto de teste independente.

O experimento evidencia a importância de avaliar o comportamento do limiar em dados não utilizados durante sua seleção, especialmente quando o objetivo é priorizar a identificação de casos positivos.

---

## ▶️ 11. Como executar

1. Escolha o notebook que deseja executar pelos botões **Open in Colab** no início deste README ou abra o arquivo correspondente no Jupyter Notebook/JupyterLab.
2. Execute as células na ordem apresentada.
3. O dataset é carregado diretamente pelo `scikit-learn` por meio de `load_breast_cancer(as_frame=True)`; não é necessário fornecer um arquivo externo.
4. No `Modelos_Unidos.ipynb`, aguarde as etapas de busca de hiperparâmetros e Nested CV, que podem demandar maior tempo computacional.
5. O `Modelo_Complementar.ipynb` realiza a calibração do limiar com previsões OOF, a seleção e avaliação do modelo, análises complementares e a comparação estatística por Nested CV.

O notebook principal salva o modelo final no arquivo `modelo_final_cancer_mama.joblib` quando a célula correspondente é executada.

---

## 📝 12. Conclusões

No experimento principal, o SVM sem K-Means foi selecionado por apresentar o maior F1-Score no conjunto de validação (0.9655), com Recall de 0.9767 — igual ao cenário com K-Means. Essa seleção pertence ao notebook principal e não deve ser confundida com a seleção realizada no experimento complementar. No conjunto de teste, os dois cenários do SVM alcançaram a mesma acurácia de 98.25%. Para o modelo final selecionado, com limiar padrão de 0,5, foram registrados precisão de 100.00%, Recall de 95.24%, F1-Score de 97.56% e ROC-AUC de 0.9957.

O experimento complementar investigou a calibração do limiar de decisão por meio de previsões OOF. Nesse cenário, o SVM com K-Means foi selecionado, utilizando limiar de aproximadamente 0,2717. No conjunto de teste, obteve acurácia de 94,74%, precisão de 89,13%, Recall de 97,62%, F1-Score de 93,18% e ROC-AUC de 0,9957. Apesar de a meta de Recall ter sido atingida durante a calibração OOF, o limiar ajustado não aumentou o Recall no teste em comparação ao limiar padrão de 0,5: ambos identificaram 41 dos 42 casos malignos. O limiar ajustado, porém, aumentou os falsos positivos de um para cinco, reduzindo a precisão, a acurácia e o F1-Score.

Na análise exploratória, o K-Means com K = 2 apresentou Silhouette Score de 0.3462, ARI de 0.6473 e NMI de 0.5246. Entretanto, a comparação por Nested CV não encontrou evidência estatisticamente significativa de diferença entre os cenários com e sem atributos derivados do K-Means. Portanto, neste conjunto de dados e sob o protocolo utilizado, não foi demonstrado ganho preditivo consistente com a inclusão dessas características.

Como limitações, destacam-se o tamanho da base (569 amostras), o conjunto de teste relativamente pequeno (114 amostras) e a ausência de validação externa em dados de outra instituição. Os resultados são experimentais e não autorizam o uso dos modelos como substitutos de avaliação clínica ou diagnóstico profissional.

---

## 📁 Repositório

O repositório público **Classificacao-Cancer-Mama-ML** reúne os notebooks `Modelos_Unidos.ipynb` e `Modelo_Complementar.ipynb`, além desta documentação. O dataset é carregado diretamente pela biblioteca `scikit-learn`.
