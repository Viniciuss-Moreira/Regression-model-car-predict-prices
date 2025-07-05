import gradio as gr
import pandas as pd
import numpy as np
import torch
import joblib
import json
import os
import traceback
from transformers import AutoTokenizer, AutoModel
from torch import nn

# --- 1. DEFINIÇÃO DA ARQUITETURA DO MODELO ---
# Colocamos a classe aqui para que a aplicação seja autocontida.
class RegressionTransformer(nn.Module):
    def __init__(self, model_name='neuralmind/bert-base-portuguese-cased'):
        super(RegressionTransformer, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)

# --- 2. CARREGAMENTO DOS ARTEFATOS GLOBAIS ---
# Esta seção é executada uma vez quando a aplicação inicia no Hugging Face Spaces.
print("Carregando artefatos do modelo...")
device = torch.device("cpu")
CACHE_DIR = "./huggingface_cache" 

try:
    # Carrega o tokenizador
    TOKENIZER_NAME = 'neuralmind/bert-base-portuguese-cased'
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, cache_dir=CACHE_DIR)
    print("Tokenizador carregado.")

    # Carrega o modelo treinado
    MODEL_PATH = './model/best_model_state.pth'
    model = RegressionTransformer(model_name=TOKENIZER_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Modelo carregado.")

    # Carrega a lista de features que o modelo espera
    FEATURES_PATH = './model/model_features.json'
    with open(FEATURES_PATH, 'r') as f:
        features_dict = json.load(f)
        model_features = features_dict['numeric'] + features_dict['categorical']
    print("Features do modelo carregadas.")

except Exception as e:
    print("!!!!!! ERRO FATAL AO CARREGAR ARTEFATOS !!!!!!")
    traceback.print_exc()
    raise e

# --- 3. FUNÇÃO DE LÓGICA / PREVISÃO ---
# Esta função contém a "inteligência" do back-end.
def predict_price(make, model_name, year, odometer, trim, body, transmission, color, interior):
    try:
        print(f"Recebida nova predição: {make}, {model_name}, {year}, {odometer}km")
        
        # Converte os inputs da interface em um DataFrame de uma linha
        input_data_dict = {
            'make': make, 'model': model_name, 'year': int(year), 'odometer': float(odometer),
            'trim': trim, 'body': body, 'transmission': transmission, 
            'color': color, 'interior': interior
        }
        input_df = pd.DataFrame([input_data_dict])
        
        # Aplica a mesma engenharia de features do treinamento
        ano_referencia = 2024
        input_df['age'] = ano_referencia - input_df['year']
        input_df['sale_month'] = 0 # Placeholder
        input_df['sale_dayofweek'] = 0 # Placeholder
        input_df['sale_dayofyear'] = 0 
        input_df['make_popularity'] = 0
        input_df['model_popularity'] = 0
        input_df['km_per_year'] = 0
        
        # Cria a "frase" textual
        def criar_representacao_textual(row):
            partes = [f"{coluna}[{str(row[coluna])}]" for coluna in model_features if coluna in row]
            return " | ".join(partes)
        text_input = input_df.apply(criar_representacao_textual, axis=1).iloc[0]
        
        # Tokeniza e prepara os tensores
        encoded_text = tokenizer.encode_plus(
            text_input, max_length=128, return_tensors='pt',
            padding='max_length', truncation=True
        )
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        
        # Faz a inferência
        with torch.no_grad():
            prediction_log = model(input_ids, attention_mask)
        
        # Pós-processa o resultado
        predicted_price = np.expm1(prediction_log.cpu().numpy()[0][0])
        
        # Formata a string de saída para o usuário
        return f"R$ {predicted_price:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    except Exception as e:
        print("!!!!!! ERRO DURANTE A PREVISÃO !!!!!!")
        traceback.print_exc()
        return "Ocorreu um erro. Verifique os logs."

# --- 4. DEFINIÇÃO DA INTERFACE COM GRADIO ---
# Esta seção define a aparência do "front-end".
with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
    gr.Markdown("# 🚗 FipeFinder AI: Previsão de Preços de Carros Usados")
    gr.Markdown("Preencha as características do veículo para receber uma estimativa de preço de mercado baseada em nosso modelo de IA. Este projeto demonstra um pipeline completo de Machine Learning, desde a análise de dados e treinamento de um modelo Transformer até o deploy de uma aplicação interativa.")
    
    with gr.Row():
        with gr.Column(scale=1):
            make_input = gr.Dropdown(label="Marca", choices=["Ford", "Chevrolet", "Honda", "Toyota", "Nissan", "Hyundai", "Kia", "BMW", "Mercedes-Benz", "Volkswagen"])
            model_input = gr.Textbox(label="Modelo", placeholder="Ex: Ka, Onix, Civic, Corolla...")
            year_input = gr.Slider(label="Ano do Modelo", minimum=2000, maximum=2024, step=1, value=2015)
            
        with gr.Column(scale=1):
            odo_input = gr.Number(label="Quilometragem (km)", value=80000)
            trim_input = gr.Textbox(label="Versão", placeholder="Ex: SE 1.0, LTZ, EXL...")
            body_input = gr.Dropdown(label="Carroceria", choices=["Sedan", "SUV", "Hatchback", "Pickup", "Minivan", "Coupe", "Wagon"])
            
        with gr.Column(scale=1):
            trans_input = gr.Radio(label="Transmissão", choices=["automatic", "manual"], value="automatic")
            color_input = gr.Textbox(label="Cor Externa", placeholder="Ex: preto, branco, prata...")
            interior_input = gr.Textbox(label="Cor Interior", placeholder="Ex: preto, cinza, bege...")
            
    predict_btn = gr.Button("Estimar Preço", variant="primary")
    output_price = gr.Label(label="Preço Estimado")

    predict_btn.click(
        fn=predict_price,
        inputs=[make_input, model_input, year_input, odo_input, trim_input, body_input, trans_input, color_input, interior_input],
        outputs=output_price
    )
    
    gr.Examples(
        examples=[
            ["Ford", "Focus", 2013, 75000.0, "SE 2.0", "Hatchback", "automatic", "prata", "preto"],
            ["Chevrolet", "Onix", 2018, 90000.0, "LT 1.0", "Hatchback", "manual", "branco", "cinza"],
            ["Toyota", "Corolla", 2020, 40000.0, "XEi", "Sedan", "automatic", "preto", "preto"],
        ],
        inputs=[make_input, model_input, year_input, odo_input, trim_input, body_input, trans_input, color_input, interior_input]
    )

# --- Lançamento da Interface ---
if __name__ == "__main__":
    demo.launch()