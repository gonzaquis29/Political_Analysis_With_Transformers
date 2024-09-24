# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# + [markdown] id="iEIK9DyrsidQ"
# ### Preprocesamiento de los discusos - técnicas de desambiguación

# + colab={"base_uri": "https://localhost:8080/"} id="dxzoI2bnsml1" outputId="37e01dc0-55ad-4f4f-90ef-ca224f5194e9"
# pip install spacy nltk scikit-learn transformers torch
# pip install es_core_news_sm
# python -m spacy download es_core_news_sm

# + id="KgZs-THosidX"
import requests
import pandas as pd

# + id="M2WMly9nsida"
import spacy


# + id="eQSkUqTnsidd"
import pandas as pd
import nltk
import re
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
import numpy as np


# + id="K3xau6LOsidf"
df_crude = pd.read_csv('../Corpus/Manifesto_corpus_estructurado.csv')

# + colab={"base_uri": "https://localhost:8080/"} id="LmIrtm7jYYVb" outputId="48bbe418-3ae5-4606-d5f3-34d35eba72bf"
indexPrueba = 4
print(df_crude['Text'][indexPrueba])
print(df_crude['Code'][indexPrueba],
df_crude['libertad_economica_score'][indexPrueba],
df_crude['libertad_personal_score'][indexPrueba])

# + colab={"base_uri": "https://localhost:8080/"} id="MFYoakalthHn" outputId="12c7cf9a-b4a7-4752-fa5a-2e25c5241b88"

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('omw-1.4')


# + id="eRc2ptsRthkH"
nlp = spacy.load("es_core_news_sm")


# + id="y8Wy_mAqNCrm"
# Las negaciones no deben ser consideradas stopwords ya que cambian el sentido de la oración.
negations = {"no", "nunca", "jamás", "nadie", "nada", "ninguno", "ninguna", "ni", "tampoco"}
stop_words = set(stopwords.words('spanish')) - negations
lemmatizer = WordNetLemmatizer()


# + id="PjNwIGoCMIKf"
def limpiar_texto(texto: str) -> str:
    # Eliminar caracteres no alfanuméricos y convertir a minúsculas
    texto = re.sub(r'[^a-záéíóúñü\s]', '', texto.lower())
    return ' '.join(texto.split())


# + id="Ax7aHRrqMpbT"
def tokenizar(texto: str) -> List[str]:
    # Tokenizar el texto en palabras
    return word_tokenize(texto)


# + id="8W0lIihLMISO"
def eliminar_stop_words(tokens: List[str]) -> List[str]:
    # Elimina palabras comunes que no aportan significado.
    return [token for token in tokens if token not in stop_words]


# + id="0IIcVYpLMz_j"
def lematizar(tokens: List[str]) -> List[str]:
    # Reduce las palabras a su forma base
    doc = nlp(' '.join(tokens))
    # Evitar lemas inválidos como "economiar"
    return [token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc]



# + id="0Sg1ilb0PF4R"
import re

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

# + id="1oPWxfM0piKx"
import torch
from transformers import pipeline
# Cargar el modelo de clasificación de sentimientos en español

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentiment_classifier = pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis", device = device)


# + id="N615gHX9nTWn"
import json
# Cargar patrones de sarcasmo desde el archivo JSON
with open('patrones_sarcasmo.json', 'r', encoding='utf-8') as f:
    sarcasm_data = json.load(f)
    sarcasm_patterns = sarcasm_data['sarcasm_patterns']
    sarcasm_keywords = sarcasm_data['sarcasm_keywords']


# + id="jbw6Enw0oALj"
def detect_sarcasm_patterns(text):
    """
    Detectar patrones de sarcasmo en una oración usando expresiones regulares.
    """
    for pattern in sarcasm_patterns:
        if re.search(pattern, text.lower()):
            return True
    return False


# + id="sY6HlKsD83nc"
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


# + id="fonLSYRjrBax"
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


# + id="EHWS9PF1rCZJ"
def remove_sarcasm(text):
    """
    Detectar y reformular sarcasmo en un texto dado.
    """
    tokens = tokenizar(text)
    if len(tokens) > 512:
        return text
    doc = nlp(text)
    sentiment = sentiment_classifier(text)[0]
    sarcasm_by_pattern = detect_sarcasm_patterns(text)
    sarcasm_by_keywords = detect_sarcasm_keywords(doc, sentiment['label'])
    if sarcasm_by_pattern or sarcasm_by_keywords:
        return reformulate_sentence(doc, sentiment['label'])
    else:
        return text


# + id="fPN3jWC9tutZ"
def preprocesar_texto(texto: str) -> Dict[str, Any]:
    texto_limpio = limpiar_texto(texto)
    texto_sin_ambiguedad = remove_sarcasm(texto_limpio)
    tokens = tokenizar(texto_sin_ambiguedad)
    tokens_sin_stop = eliminar_stop_words(tokens)
    tokens_lematizados = lematizar(tokens_sin_stop)

    return ' '.join(tokens_lematizados)


# + colab={"base_uri": "https://localhost:8080/", "height": 36} id="WaZy5-EPtw7U" outputId="898c1583-596d-4e89-98a6-a91e6745c7bc"
texto = df_crude.iloc[10, 0]
texto = "Que buen trabajo que estas haciendo!. Sigue así que solo lo empeoras más"
texto_preprocesado = preprocesar_texto(texto)
texto_preprocesado

# + colab={"base_uri": "https://localhost:8080/"} id="1-xvAZdzZ7aI" outputId="7452e092-56d6-421f-d604-a459c514d4e3"
print(df_crude['libertad_economica_score'].value_counts())

# + colab={"base_uri": "https://localhost:8080/"} id="f1KN7fDqaBSx" outputId="bccc39fc-05b8-4609-dba7-31f65a85b302"
print(df_crude['libertad_personal_score'].value_counts())


# + colab={"base_uri": "https://localhost:8080/"} id="LFCxu2Y1bRBl" outputId="0eff0229-bf74-4e92-f2a5-2fc4970e362b"

# Función para obtener un subconjunto balanceado
def get_balanced_subset(df, column, n_per_category=50):
    balanced_df = pd.DataFrame()
    for category in [-1, 0, 1]:
        category_df = df[df[column] == category]
        if len(category_df) > n_per_category:
            category_df = category_df.sample(n=n_per_category, random_state=42)
        balanced_df = pd.concat([balanced_df, category_df])
    return balanced_df

# Obtener subconjuntos balanceados para ambas columnas
df_economica = get_balanced_subset(df_crude, 'libertad_economica_score')
df_personal = get_balanced_subset(df_crude, 'libertad_personal_score')

# Combinar los subconjuntos
df_balanced = pd.concat([df_economica, df_personal]).drop_duplicates()

# Verificar los recuentos
print(df_balanced['libertad_economica_score'].value_counts())
print(df_balanced['libertad_personal_score'].value_counts())

# + colab={"base_uri": "https://localhost:8080/", "height": 423} id="h67PoQoQbgAA" outputId="071a640c-b8ef-42fd-f7d2-194d3346caae"
df_balanced

# + id="nPcVjGai9c7j"


#df_crude['text_processed'] = df_crude['Text'].apply(preprocesar_texto)



# Función para procesar y guardar por batches
def procesar_por_batches(df, batch_size, output_prefix):
    num_batches = int(np.ceil(len(df) / batch_size))
    for i in range(num_batches):
        if i <= 58:
            continue
        batch = df[i * batch_size : (i + 1) * batch_size].copy()  # Obtener batch
        batch['text_processed'] = batch['Text'].apply(preprocesar_texto)  # Procesar batch
        batch.to_csv(f'results/{output_prefix}_batch_{i + 1}.csv', index=False)  # Guardar batch procesado
        print(f'Batch {i + 1}/{num_batches} procesado y guardado.')
        if i == 59:
            break


# Cargar tu DataFrame
# df_crude = pd.read_csv('tu_dataframe.csv')

# Parámetros
batch_size = 5000
output_prefix = 'resultados_parciales'

# Procesar por batches y guardar resultados
procesar_por_batches(df_crude, batch_size, output_prefix)



# + id="esi78l1ZmzGP"
df_crude.to_csv('df_balanced_3.csv', index=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 423} id="9H-4_F22l-So" outputId="29c40aed-fd84-4093-f862-bfb9a62b46a1"

