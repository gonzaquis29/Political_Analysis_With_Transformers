
# Preprocesamiento de los discusos - técnicas de desambiguación

# pip install spacy nltk scikit-learn transformers torch
# pip install es_core_news_sm
# python -m spacy download es_core_news_sm


import requests
import pandas as pd
import numpy as np
import spacy
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split


df_crude = pd.read_csv('../extraction/structured/reddit_posts_estructurado.csv')
df_crude['text'] = df_crude['text'].astype(str)

# Pruebas
indexPrueba = 4
print(df_crude['text'][indexPrueba], df_crude['libertad_economica_score'][indexPrueba],
df_crude['libertad_personal_score'][indexPrueba] )

# Cargar las librerías de nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('omw-1.4')


# Cargar corpus de palabras de español
nlp = spacy.load("es_core_news_sm")


# Las negaciones no deben ser consideradas stopwords ya que cambian el sentido de la oración.
negations = {"no", "nunca", "jamás", "nadie", "nada", "ninguno", "ninguna", "ni", "tampoco"}
stop_words = set(stopwords.words('spanish')) - negations
lemmatizer = WordNetLemmatizer()

# Funciones auxiliares

def limpiar_texto(texto: str) -> str:
    # Eliminar caracteres no alfanuméricos y convertir a minúsculas
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto.lower())
    return ' '.join(texto.split())


def tokenizar(texto: str) -> List[str]:
    # Tokenizar el texto en palabras
    return word_tokenize(texto)

def eliminar_stop_words(tokens: List[str]) -> List[str]:
    # Elimina palabras comunes que no aportan significado.
    return [token for token in tokens if token not in stop_words]

def lematizar(tokens: List[str]) -> List[str]:
    # Reduce las palabras a su forma base
    doc = nlp(' '.join(tokens))
    # Evitar lemas inválidos como "economiar"
    return [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc]

def desambiguar(texto: str) -> str:

    #Función que elimina indicios de ambigüedad, sarcasmo, ironía o figuras retóricas,
    #transformándolos en una versión más neutra sin perder el significado.

    # Diccionario de frases irónicas/sarcásticas y sus versiones neutras
    sarcasmo_neutro = {
        r"\bsí, claro\b": "obviamente no",       # "sí, claro" en tono irónico
        r"\bpor supuesto\b": "de ninguna manera", # "por supuesto" irónico
        r"\bqué sorpresa\b": "esto era esperado", # "qué sorpresa" sarcástico
        r"\bestoy seguro\b": "no estoy seguro",   # "estoy seguro" irónico
        r"\bno me digas\b": "es obvio",           # "no me digas" irónico
        r"\bqué gran trabajo\b": "mal trabajo",   # "qué gran trabajo" irónico
    }

    # Reemplazar frases irónicas por versiones neutras
    for patron, reemplazo in sarcasmo_neutro.items():
        texto = re.sub(patron, reemplazo, texto, flags=re.IGNORECASE)

    # Eliminar uso excesivo de signos de exclamación o interrogación
    texto = re.sub(r'[!?]+', '.', texto)

    # Eliminar comillas que pueden implicar ironía
    texto = re.sub(r'[\'\"]', '', texto)

    # Eliminar exageraciones comunes y normalizar las frases
    exageraciones = ['muy', 'sumamente', 'increíblemente', 'extremadamente', 'totalmente']
    for exag in exageraciones:
        texto = re.sub(rf'\b{exag}\b', '', texto, flags=re.IGNORECASE)

    # Eliminar frases que inviten a interpretación figurativa o sarcástica
    texto = re.sub(r"\b(debería ser)\b", "", texto)

    return texto.strip()

import torch
from transformers import pipeline
# Cargar el modelo de clasificación de sentimientos en español

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentiment_classifier = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis", device = device)


import json
# Cargar patrones de sarcasmo desde el archivo JSON
with open('patrones_sarcasmo.json', 'r', encoding='utf-8') as f:
    sarcasm_data = json.load(f)
    sarcasm_patterns = sarcasm_data['sarcasm_patterns']
    sarcasm_keywords = sarcasm_data['sarcasm_keywords']


def detect_sarcasm_patterns(text):
    """
    Detectar patrones de sarcasmo en una oración usando expresiones regulares.
    """
    for pattern in sarcasm_patterns:
        if re.search(pattern, text.lower()):
            return True
    return False


def detect_sarcasm_keywords(doc, sentiment_label):
    """
    Detectar sarcasmo basado en palabras clave y el sentimiento.
    """
    for token in doc:
        if sentiment_label == 'NEG' and token.text.lower() in sarcasm_keywords['positivas']:
            return True
        if sentiment_label == 'POS' and token.text.lower() in sarcasm_keywords['negativas']:
            return True
    return False


def reformulate_sentence(doc, sentiment_label):
    """
    Reformular una oración detectada como sarcástica.
    """
    reformulated = []
    sarcasm_detected = False

    for token in doc:
        if token.text.lower() in sarcasm_keywords["positivas"] and (sentiment_label == 'NEG' or detect_sarcasm_patterns(doc.text)):
            reformulated.append("no es " + token.text)
            sarcasm_detected = True
        elif token.text.lower() in sarcasm_keywords["negativas"] and (sentiment_label == 'POS' or detect_sarcasm_patterns(doc.text)):
            reformulated.append("no es tan " + token.text)
            sarcasm_detected = True
        else:
            reformulated.append(token.text)

    if sarcasm_detected:
        result = " ".join(reformulated)
        # Reformulaciones adicionales para sarcasmo común
        result = re.sub(r"(?i)\bclaro que sí\b", "no es cierto que", result)
        result = re.sub(r"(?i)\bqué gran\b", "no es tan gran", result)
        result = re.sub(r"(?i)\bobviamente\b", "no necesariamente", result)
        result = re.sub(r"(?i)\bseguramente\b", "posiblemente no", result)
        return result
    else:
        return doc.text


def remove_sarcasm(text):
    """
    Detectar y reformular sarcasmo en un texto dado.
    """
    tokens = tokenizar(text)
    if len(tokens) > 512:
        print("Supera los tokens: ",len(tokens))
        print("Este es el texto",text)
        tokensTrunc = tokens[:512]
        return ' '.join(tokensTrunc)

    doc = nlp(text)
    sentiment = sentiment_classifier(text)[0]
    sarcasm_by_pattern = detect_sarcasm_patterns(text)
    sarcasm_by_keywords = detect_sarcasm_keywords(doc, sentiment['label'])
    if sarcasm_by_pattern or sarcasm_by_keywords:
        return reformulate_sentence(doc, sentiment['label'])
    else:
        return text


def preprocesar_texto(texto: str) -> Dict[str, Any]:
    texto_limpio = limpiar_texto(texto)
    tokensAntes = tokenizar(texto_limpio)
    if len(tokensAntes) < 300:
        texto_sin_ambiguedad = remove_sarcasm(texto_limpio)
    else:
        print("No se hace proceso de desambiguacion")
        texto_sin_ambiguedad = texto_limpio

    tokens = tokenizar(texto_sin_ambiguedad)
    tokens_sin_stop = eliminar_stop_words(tokens)
    tokens_lematizados = lematizar(tokens_sin_stop)

    return ' '.join(tokens_lematizados)


texto = df_crude.iloc[10, 0]
texto = "Que buen trabajo que estas haciendo!. Sigue así que solo lo empeoras más"
texto_preprocesado = preprocesar_texto(texto)
texto_preprocesado

print(df_crude['libertad_economica_score'].value_counts())

print(df_crude['libertad_personal_score'].value_counts())



df_crude['text_processed'] = df_crude['text'].apply(preprocesar_texto)
df_crude.to_csv("reddit_preprocessed.csv")

folder = 'results_ideo'

# Función para procesar y guardar por batches
def procesar_por_batches(df, batch_size, output_prefix):
    num_batches = int(np.ceil(len(df) / batch_size))
    for i in range(num_batches):
        batch = df[i * batch_size : (i + 1) * batch_size].copy()  # Obtener batch
        batch['text_processed'] = batch['text'].apply(preprocesar_texto)  # Procesar batch
        batch.to_csv(f'{folder}/{output_prefix}_batch_{i + 1}.csv', index=False)  # Guardar batch procesado
        print(f'Batch {i + 1}/{num_batches} procesado y guardado.')


# Cargar tu DataFrame
# df_crude = pd.read_csv('tu_dataframe.csv')

# Parámetros
batch_size = 1000
output_prefix = 'resultados_parciales'

# Procesar por batches y guardar resultados
#procesar_por_batches(df_crude, batch_size, output_prefix)

#df_crude.to_csv('df_balanced_3.csv', index=False)
