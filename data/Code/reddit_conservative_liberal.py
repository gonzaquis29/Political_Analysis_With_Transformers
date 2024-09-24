#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[6]:


reddit_politics = pd.read_csv("../Raw/reddit_dataset.csv")
reddit_politics.head()


# In[7]:


reddit_politics_shuffled = reddit_politics.sample(frac=1, random_state=42)


# In[8]:


reddit_politics_shuffled.head()


# In[10]:


reddit_politics_shuffled = reddit_politics_shuffled.drop("Id",axis = 1)


# In[12]:


reddit_politics_shuffled = reddit_politics_shuffled.drop("Score",axis = 1)
reddit_politics_shuffled = reddit_politics_shuffled.drop("URL",axis = 1)
reddit_politics_shuffled = reddit_politics_shuffled.drop("Num of Comments",axis = 1)
reddit_politics_shuffled = reddit_politics_shuffled.drop("Date Created",axis = 1)


# In[13]:


reddit_politics_shuffled


# In[25]:


reddit_politics_shuffled["Text"].to_list()[3]


# In[35]:


reddit_politics_shuffled["TextComplete"] = reddit_politics_shuffled["Title"] + " " +reddit_politics_shuffled["Text"].fillna("")


# In[27]:


from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to('cuda')

# In[29]:


# Oración de prueba
sentence = "This is a cat."

# Traducir la oración
inputs = tokenizer([sentence], return_tensors="pt", padding=True).to('cuda')
translated = model.generate(**inputs)
translated_sentence = tokenizer.decode(translated[0], skip_special_tokens=True)


# In[32]:


print(translated_sentence)


# In[33]:


def translate_batch(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to('cuda')
    translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


# In[28]:


# Cargar el modelo y el tokenizador para inglés a español
model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


# In[37]:


def translate_batch(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


# In[40]:


def save_translation(batch_number, translated_batch):
    output_file = "traduccion_reddit/translated_batch_"+ str(batch_number) + ".csv"
    df_translated = pd.DataFrame(translated_batch, columns=['translated_statement'])
    df_translated.to_csv(output_file, index=False)
    print(f"Batch {batch_number} guardado en {output_file}")


# In[44]:


# Traducir la columna 'statement' por lotes para no sobrecargar la memoria
# Preparar batches
batch_size = 10
batches = [(i//batch_size, reddit_politics_shuffled['TextComplete'][i:i+batch_size].tolist()) for i in range(0, len(reddit_politics_shuffled), batch_size)]


# In[45]:


def process_batch(batch_data):
    batch_number, batch = batch_data
    translated_batch = translate_batch(batch)
    save_translation(batch_number, translated_batch)


for batch_data in batches:
    batch_number, batch = batch_data
    translated_batch = translate_batch(batch)
    save_translation(batch_number, translated_batch)
    break

