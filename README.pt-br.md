[🇬🇧 View in English](./README.md)
---

# 🚗 Modelo de Regressão: Previsão de Preços de Carros com Deep Learning

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.29-FF7622?style=for-the-badge&logo=gradio&logoColor=white)](https://www.gradio.app/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=for-the-badge)](https://huggingface.co/spaces)

**Link para a Aplicação ao Vivo:** [**➡️ Clique Aqui para Testar a Demo ao Vivo**](https://vinimoreira-regression-prices-cars-b4-it.hf.space)
---
* **[🔬 Ver o Notebook de Treinamento Completo no GitHub](./notebooks/notebook.ipynb)** * **[🚀 Abrir e Executar Interativamente no Google Colab](https://colab.research.google.com/drive/1r--GE8Np_mnvnmFqMAOKteCkH2Pe0HwE?usp=sharing)**
---
Este projeto apresenta um fluxo de trabalho completo de Machine Learning, desde a análise de dados até o fine-tuning e o deploy de um modelo de deep learning para regressão tabular. É uma aplicação interativa para prever o preço de venda de carros usados no mercado brasileiro.

![App Demo](./img/demo.gif)
---

## 📖 Descrição do Projeto

Este projeto apresenta o **FipeFinder AI**, uma aplicação de machine learning projetada para prever o preço de venda de carros usados no mercado brasileiro. Diferente de tabelas de preço estáticas, este modelo utiliza um grande dataset do mundo real com mais de 550.000 anúncios de veículos para capturar as nuances e relações complexas que determinam o verdadeiro valor de mercado de um carro.

O núcleo do projeto é um modelo de regressão que passou por fine-tuning, demonstrando uma abordagem avançada para regressão com dados tabulares. Todo o processo, da exploração de dados à aplicação web final, está documentado aqui.

---

## 🛠️ Fluxo de Trabalho & Arquitetura do Projeto

O projeto segue um ciclo de vida clássico e rigoroso de Ciência de Dados:
1.  **Análise e Limpeza de Dados (EDA):** A fase inicial envolveu uma análise estatística profunda do dataset para entender suas distribuições, correlações e problemas de qualidade. Valores nulos foram tratados com imputação estratégica (moda para features categóricas, mediana para numéricas).
2.  **Engenharia de Features Avançada:** Novas features de alto valor foram criadas a partir dos dados brutos para melhorar o poder preditivo do modelo. Isso inclui o cálculo da `idade` do veículo, sua taxa de uso (`km_por_ano`) e a `popularidade` de sua marca e modelo.
3.  **Engenharia e Fine-Tuning de Modelo:** Esta é a essência da engenharia de IA do projeto.
    * Um modelo de linguagem pré-treinado para o português, `neuralmind/bert-base-portuguese-cased` (BERTimbau), foi usado como o "backbone".
    * Uma "cabeça" de regressão customizada (composta por camadas Lineares e de Dropout) foi adicionada ao topo do modelo BERT.
    * O modelo inteiro passou por **fine-tuning** nos dados pré-processados. O loop de treinamento foi construído do zero usando PyTorch, implementando técnicas como agendamento de taxa de aprendizado e salvamento do melhor modelo com base na perda de validação para prevenir overfitting.
    * Todo o processo de treinamento foi acelerado usando uma GPU NVIDIA local (CUDA), reduzindo o tempo de treinamento de horas para minutos.
4.  **Avaliação:** O modelo final foi avaliado em um conjunto de teste separado (com mais de 110.000 amostras nunca vistas), alcançando um **Erro Médio Absoluto (MAE) de aproximadamente R$ 1.721**.
5.  **Deploy:** O modelo treinado e os passos de pré-processamento foram empacotados em uma aplicação web interativa usando **Gradio** e implantados no **Hugging Face Spaces** para acesso público.

---

## ✨ Destaques Técnicos

* **Transformer para Dados Tabulares:** Foi aplicado um modelo de linguagem de ponta a um problema de regressão tabular estruturada, convertendo features em uma sentença descritiva.
* **Fine-Tuning de Ponta a Ponta:** Demonstrou o processo completo de adaptação de um modelo pré-treinado para uma nova tarefa, incluindo a construção da arquitetura customizada e do loop de treinamento em PyTorch.
* **Treinamento Acelerado por GPU:** Foi configurado e utilizou com sucesso uma GPU local com CUDA para treinar um modelo de deep learning em um dataset grande.
* **Análise de Dados Aprofundada:** Foram exibidos sólidos fundamentos de ciência de dados através de uma rigorosa análise exploratória e engenharia de features.
* **Demo de ML Interativa:** Foi implantado o modelo final como uma aplicação Gradio amigável, tornando os resultados acessíveis e demonstráveis.

---

## 🚀 Stack de Tecnologias

* **Ciência de Dados & ML:** Pandas, NumPy, Scikit-learn, PyTorch, Hugging Face Transformers
* **UI & Deploy:** Gradio, Hugging Face Spaces, Git LFS

---

## ⚙️ Como Executar Localmente

Este projeto tem dois componentes principais: a aplicação final com Gradio e o notebook de treinamento.

### 1. Executando a Aplicação Final (Demo com Gradio) Localmente

Estas instruções são para rodar o modelo pré-treinado na interface interativa em sua máquina local.

1.  **Pré-requisitos:** Python 3.11+, Git, e Git LFS.
2.  **Clonar o Repositório:**
    ```bash
    git clone [https://github.com/Viniciuss-Moreira/Regression-model-car-predict-prices.git](https://github.com/Viniciuss-Moreira/Regression-model-car-predict-prices.git)
    cd Regression-model-car-predict-prices
    ```
3.  **Configurar o Ambiente Virtual:**
    ```bash
    python -m venv .venv
    # No Windows:
    .\.venv\Scripts\activate
    # No Mac/Linux:
    source .venv/bin/activate
    ```
4.  **Instalar as Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Executar a Aplicação:**
    ```bash
    python app.py
    ```
    Isso iniciará a interface Gradio em uma URL local (ex: `http://127.0.0.1:7860`).

### 2. Replicando o Treinamento do Modelo

O processo completo de análise de dados, engenharia de features e treinamento do modelo está documentado no Jupyter Notebook. Para replicá-lo, você precisará de uma máquina com uma GPU NVIDIA compatível com CUDA e um ambiente Python/CUDA configurado corretamente.

## 🔮 Melhorias Futuras

* **Ajuste de Hiperparâmetros:** Usar uma abordagem sistemática como Optuna ou Ray Tune para encontrar a combinação ótima de taxa de aprendizado, tamanho do lote e arquitetura de rede.
* **IA Explicável (XAI):** Implementar técnicas como SHAP para entender *quais* features (marca, idade, km) estão influenciando mais as previsões do modelo.
* **Engenharia de Features Avançada:** Incorporar dados externos, como indicadores econômicos (inflação, juros) da época da venda, para avaliar se melhoram a precisão da previsão.
