# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

FILE_NAME = "poli_bias"

data_corpus_pre = pd.read_csv('../../../data/corpus/political_bias_preprocessed.csv')
#data_corpus = pd.read_csv('manifesto_preprocessed.csv')
#data_corpus_pre = data_corpus_pre.sample(n=10000, random_state=42)

data_corpus_pre['text_processed'] = data_corpus_pre['text_processed'].astype(str)

from sklearn.model_selection import train_test_split

train_df, eval_df = train_test_split(data_corpus_pre, test_size=0.2, random_state=42)

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
# Definir el modelo
class LibertyPredictor(nn.Module):
    def __init__(self, pretrained_model_name):
        super(LibertyPredictor, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc_personal = nn.Linear(self.bert.config.hidden_size, 3)  # 3 clases para libertad personal
        self.fc_economic = nn.Linear(self.bert.config.hidden_size, 3)  # 3 clases para libertad económica

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        x = self.dropout(pooled_output)
        personal_liberty = self.fc_personal(x)
        economic_liberty = self.fc_economic(x)
        return personal_liberty, economic_liberty

# Definir el dataset personalizado
class LibertyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text_processed
        self.targets = self._process_targets(dataframe[['libertad_personal_score', 'libertad_economica_score']].values)
        self.max_len = max_len

    def _process_targets(self, scores):
        def score_to_class(score):
            if score < -0.33:
                return 0  # -1
            elif score < 0.33:
                return 1  # 0
            else:
                return 2  # 1

        return [[score_to_class(personal), score_to_class(economic)] for personal, economic in scores]


    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

# Configurar hiperparámetros
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-04
PRETRAINED_MODEL = "dccuchile/bert-base-spanish-wwm-uncased"  # Cambiar a DistilBERT
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

# Arreglar los indices
train_df = train_df.reset_index(drop=True)
eval_df = eval_df.reset_index(drop=True)
# Crear datasets y dataloaders
train_dataset = LibertyDataset(train_df, tokenizer, MAX_LEN)
eval_dataset = LibertyDataset(eval_df, tokenizer, MAX_LEN)

train_dataset.__len__()

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
eval_loader = DataLoader(eval_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=1)

train_dataset.data

# Inicializar el modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LibertyPredictor(PRETRAINED_MODEL)
model = model.to(device)


# Congelar las capas de BERT excepto las dos últimas
for name, param in model.bert.named_parameters():
    if 'layer.4' in name or 'layer.5' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


# Definir el optimizador y la función de pérdida
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Función de entrenamiento
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for _, data in enumerate(train_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        personal_liberty, economic_liberty = model(ids, mask)
        loss = criterion(personal_liberty, targets[:, 0]) + criterion(economic_liberty, targets[:, 1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Función de evaluación
def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_personal_preds = []
    all_economic_preds = []
    all_personal_targets = []
    all_economic_targets = []

    with torch.no_grad():
        for _, data in enumerate(eval_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)

            personal_liberty, economic_liberty = model(ids, mask)
            loss = criterion(personal_liberty, targets[:, 0]) + criterion(economic_liberty, targets[:, 1])
            total_loss += loss.item()

            all_personal_preds.extend(torch.argmax(personal_liberty, dim=1).cpu().numpy())
            all_economic_preds.extend(torch.argmax(economic_liberty, dim=1).cpu().numpy())
            all_personal_targets.extend(targets[:, 0].cpu().numpy())
            all_economic_targets.extend(targets[:, 1].cpu().numpy())

    avg_loss = total_loss / len(eval_loader)
    personal_f1 = f1_score(all_personal_targets, all_personal_preds, average='weighted')
    economic_f1 = f1_score(all_economic_targets, all_economic_preds, average='weighted')

    return avg_loss, personal_f1, economic_f1

# Entrenamiento del modelo

with open('training_logs_'+FILE_NAME+'.txt', 'a') as log_file:
    for epoch in range(EPOCHS):
        train(model, train_loader, optimizer, criterion, device)
        eval_loss, personal_f1, economic_f1 = evaluate(model, eval_loader, criterion, device)
        train_loss, train_personal_f1, train_economic_f1 = evaluate(model, train_loader, criterion, device)
            
        log_file.write(f'Epoch {epoch+1}/{EPOCHS}, '
                        f'Train Personal F1: {train_personal_f1:.4f}, Train Economic F1: {train_economic_f1:.4f}, '
                        f'Validation Loss: {eval_loss:.4f}, '
                        f'Validation Personal F1: {personal_f1:.4f}, Validation Economic F1: {economic_f1:.4f}\n')

        print(f'Epoch {epoch+1}/{EPOCHS}, Validation Loss: {eval_loss:.4f}, '
                f'Personal Liberty F1: {personal_f1:.4f}, Economic Liberty F1: {economic_f1:.4f}')
# Guardar el modelo
MODEL_NAME = FILE_NAME + '_liberty_predictor.pth'
torch.save(model.state_dict(), MODEL_NAME)

print("Entrenamiento completado y modelo guardado.")



# Función para hacer predicciones
def predict(model, text, tokenizer, max_len, device):
    model.eval()
    encoded_text = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    with torch.no_grad():
        personal_liberty, economic_liberty = model(input_ids, attention_mask)

    personal_score = torch.argmax(personal_liberty, dim=1).item() - 1
    economic_score = torch.argmax(economic_liberty, dim=1).item() - 1
    return personal_score, economic_score

# Ejemplo de uso
texto_ejemplo = "conservador"
libertad_personal, libertad_economica = predict(model, texto_ejemplo, tokenizer, MAX_LEN, device)
print(f"Libertad personal: {libertad_personal:.2f}")
print(f"Libertad económica: {libertad_economica:.2f}")
