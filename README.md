# Projeto: Previsão de Doenças Cardíacas

## Visão Geral

Este projeto tem como objetivo desenvolver um modelo de classificação capaz de prever a presença de doença cardíaca em pacientes com base em características clínicas e exames médicos. O modelo utiliza técnicas de Machine Learning para auxiliar profissionais de saúde na identificação precoce de riscos cardiovasculares.

## Estrutura do Projeto

```
n3-ciencia-dados/
├── README.md
├── requirements.txt
├── modelo_final.pkl
├── scaler.pkl
├── notebooks/
│   └── 02_modelagem.ipynb
├── data/
│   └── heart_disease.csv
└── scripts/
    └── predict.py
```

## Como Executar

### 1. Instalação das Dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o Notebook de Modelagem

Abra o Jupyter Notebook e execute:

```bash
jupyter notebook notebooks/02_modelagem.ipynb
```

O notebook irá:
- Carregar e explorar os dados
- Treinar três modelos diferentes
- Avaliar e comparar os modelos
- Salvar o melhor modelo em `modelo_final.pkl`

### 3. Usar o Modelo para Previsões

Após treinar o modelo, você pode usar o script de predição:

```bash
python scripts/predict.py
```

Ou passar parâmetros customizados:

```bash
python scripts/predict.py 55 1 2 140 260 0 0 150 1 2.5 1 1 3
```

## Parte 1: A Fundação do Projeto - O Problema de Negócio

### 1.1. Domínio do Problema

As doenças cardiovasculares são a principal causa de morte no mundo, responsáveis por milhões de óbitos anualmente. A identificação precoce de fatores de risco e a predição da probabilidade de desenvolvimento de doenças cardíacas são fundamentais para a prevenção e tratamento adequado.

Este projeto se insere no contexto da saúde pública e medicina preventiva, onde a análise de dados clínicos pode auxiliar profissionais de saúde a tomar decisões mais informadas sobre o diagnóstico e tratamento de pacientes.

### 1.2. Pergunta de Negócio

A pergunta central que buscamos responder foi: **"Quais características clínicas e exames médicos têm maior impacto na predição de doenças cardíacas, e é possível construir um modelo confiável para identificar pacientes com risco de desenvolver problemas cardiovasculares?"**

### 1.3. Objetivo do Modelo

O objetivo foi construir um modelo de classificação capaz de prever a presença de doença cardíaca em pacientes com base em características como idade, sexo, tipo de dor no peito, pressão arterial, colesterol, resultados de exames e outros indicadores clínicos. O modelo fornece uma ferramenta de apoio para profissionais de saúde na avaliação de riscos cardiovasculares.

## Parte 2: A Jornada dos Dados - Pipeline e Arquitetura

### 2.1. Origem e Repositório de Dados

**Fonte Original:** O dataset utilizado foi obtido do Kaggle, uma plataforma amplamente utilizada para compartilhamento de datasets de ciência de dados. O dataset contém informações clínicas de pacientes relacionadas a doenças cardíacas.

**Arquitetura de Armazenamento:** Para este projeto, adotamos uma arquitetura simples e eficiente:

- **Data Lake (Dados Brutos):** O arquivo CSV original (`heart_disease.csv`) é armazenado na pasta `/data`, representando os dados brutos sem processamento.
- **Data Warehouse (Dados Processados):** Os dados processados e transformados são gerados durante a execução dos notebooks e mantidos em memória durante o processamento.
- **Modelo Persistido:** O modelo treinado e o scaler são salvos em arquivos `.pkl` na raiz do projeto para reutilização.

Esta arquitetura foi escolhida por ser adequada para projetos de médio porte, permitindo versionamento dos dados brutos, reprodutibilidade do processamento e facilidade de deploy do modelo final.

### 2.2. Pipeline de Dados

#### Ingestão

Os dados foram coletados do Kaggle e armazenados localmente como arquivo CSV na pasta `/data`. O dataset contém 13 features (variáveis preditoras) e 1 target (variável a ser prevista).

#### Limpeza e Transformação (ETL/ELT)

As principais etapas de limpeza e preparação realizadas foram:

1. **Verificação de Valores Ausentes:** Verificamos a presença de valores nulos no dataset. O dataset utilizado não apresentou valores ausentes.
2. **Verificação de Duplicatas:** Verificamos e removemos registros duplicados, se houvessem.
3. **Padronização de Dados:** Todas as variáveis já estavam em formato numérico, facilitando o processamento.
4. **Normalização:** Aplicamos StandardScaler para normalizar as features, garantindo que todas as variáveis tenham a mesma escala e evitando que variáveis com valores maiores dominem o modelo.

#### Análise Exploratória (EDA)

A análise exploratória foi fundamental para:
- Entender a distribuição das variáveis
- Identificar correlações entre features e o target
- Verificar o balanceamento da classe target
- Identificar possíveis outliers

Os insights da EDA ajudaram a validar a escolha das features e a entender melhor o comportamento dos dados.

#### Preparação para Modelagem

A etapa final de preparação incluiu:

1. **Separação de Features e Target:** Separamos as variáveis preditoras (X) da variável target (y).
2. **Divisão Treino/Teste:** Dividimos os dados em conjunto de treino (80%) e teste (20%), utilizando estratificação para manter a proporção das classes.
3. **Normalização:** Aplicamos StandardScaler nos dados de treino e teste, garantindo que o modelo receba dados na mesma escala.

## Parte 3: O Coração do Projeto - Modelagem e Avaliação Comparativa

### 3.1. Treinamento de Três Modelos

Foram treinados três algoritmos de classificação:

1. **Regressão Logística:** Modelo linear que estima a probabilidade de uma classe usando uma função logística. É interpretável e funciona bem como baseline.
2. **Árvore de Decisão:** Modelo não-paramétrico que cria regras de decisão baseadas nas features. É fácil de interpretar e não requer normalização prévia.
3. **Random Forest:** Ensemble de árvores de decisão que combina múltiplas árvores para melhorar a performance e reduzir overfitting.

### 3.2. Avaliação com Três Métricas

Foram utilizadas quatro métricas para avaliar os modelos:

1. **Acurácia:** Mede a proporção de previsões corretas em relação ao total. É uma métrica geral de desempenho, mas pode ser enganosa em datasets desbalanceados.

2. **Precisão:** Mede a proporção de casos positivos previstos que são realmente positivos. É importante quando queremos minimizar falsos positivos (diagnosticar doença quando não há).

3. **Recall:** Mede a proporção de casos positivos reais que foram corretamente identificados. É crucial em problemas de saúde, pois queremos identificar o máximo de casos reais de doença, mesmo que isso gere alguns falsos positivos.

4. **F1-Score:** Média harmônica entre Precisão e Recall. Fornece um equilíbrio entre as duas métricas e é útil quando precisamos de uma métrica única que considere ambos os aspectos.

Para problemas de saúde como predição de doenças cardíacas, o **Recall é especialmente importante**, pois é melhor ter alguns falsos positivos do que perder casos reais de doença que poderiam ser tratados.

### 3.3. Análise Comparativa dos Resultados

Os resultados dos três modelos foram comparados em uma tabela, avaliando cada métrica. O modelo com melhor F1-Score foi selecionado como modelo final, pois essa métrica oferece um equilíbrio entre Precisão e Recall, ambos importantes para o problema em questão.

**Modelo Escolhido:** O modelo selecionado foi aquele que apresentou o melhor F1-Score, garantindo um bom equilíbrio entre identificar corretamente os casos de doença (Recall) e minimizar falsos positivos (Precisão).

## Parte 4: Tornando o Modelo Útil - Deploy

### 4.1. Salvando o Modelo Treinado

Após a seleção do melhor modelo, ele foi salvo usando a biblioteca `joblib`:

```python
joblib.dump(best_model, '../modelo_final.pkl')
joblib.dump(scaler, '../scaler.pkl')
```

O scaler também foi salvo para garantir que novos dados sejam normalizados da mesma forma que os dados de treino.

### 4.2. Carregando e Utilizando o Modelo

O modelo pode ser carregado e utilizado para fazer previsões em novos pacientes:

```python
loaded_model = joblib.load('../modelo_final.pkl')
loaded_scaler = joblib.load('../scaler.pkl')

novo_paciente = {
    'age': 55,
    'sex': 1,
    'chest_pain_type': 2,
    'resting_bp': 140,
    'cholesterol': 260,
    'fasting_bs': 0,
    'resting_ecg': 0,
    'max_hr': 150,
    'exercise_angina': 1,
    'oldpeak': 2.5,
    'st_slope': 1,
    'num_vessels': 1,
    'thalassemia': 3
}

novo_paciente_df = pd.DataFrame([novo_paciente])
novo_paciente_scaled = loaded_scaler.transform(novo_paciente_df)
previsao = loaded_model.predict(novo_paciente_scaled)
probabilidade = loaded_model.predict_proba(novo_paciente_scaled)
```

O modelo retorna tanto a classe prevista (0 ou 1) quanto as probabilidades associadas, permitindo uma avaliação mais detalhada do risco do paciente.

## Descrição das Features

- **age:** Idade do paciente
- **sex:** Sexo (0 = Feminino, 1 = Masculino)
- **chest_pain_type:** Tipo de dor no peito (0-3)
- **resting_bp:** Pressão arterial em repouso (mm Hg)
- **cholesterol:** Colesterol sérico (mg/dl)
- **fasting_bs:** Açúcar no sangue em jejum > 120 mg/dl (0 = Não, 1 = Sim)
- **resting_ecg:** Resultados eletrocardiográficos em repouso (0-2)
- **max_hr:** Frequência cardíaca máxima alcançada
- **exercise_angina:** Angina induzida por exercício (0 = Não, 1 = Sim)
- **oldpeak:** Depressão do ST induzida por exercício
- **st_slope:** Inclinação do segmento ST de pico do exercício (0-2)
- **num_vessels:** Número de vasos principais coloridos por fluoroscopia (0-3)
- **thalassemia:** Tipo de talassemia (0-3)
- **target:** Presença de doença cardíaca (0 = Não, 1 = Sim)

## Tecnologias Utilizadas

- Python 3.8+
- pandas: Manipulação de dados
- numpy: Operações numéricas
- scikit-learn: Machine Learning
- matplotlib/seaborn: Visualizações
- joblib: Serialização de modelos
- jupyter: Notebooks interativos

## Autores

Projeto desenvolvido para o Trabalho Final de Ciência de Dados - N3.

## Licença

Este projeto é apenas para fins educacionais.
