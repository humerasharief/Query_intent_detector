# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 07:53:13 2021

@author: Winston Fernandes
"""

import streamlit as st
import pandas as pd
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
import re
import matplotlib.pyplot as plt
import string
import tensorflow as tf
import tensorflow_addons as tfa
import pickle as pkl
from tensorflow.keras.preprocessing import sequence
import numpy as np
import seaborn as sns
import nltk
nltk.download('stopwords')
import plotly.express as px
import altair as alt

st.title('Bank query intention detector')
df=pd.read_csv('bank_test.csv',encoding='cp1252')
selected_text = st.selectbox('Select the test data example:', df.text.values)

with open("bank_classes.pkl", "rb") as f:
    bank_classes = pkl.load(f)
    
def preprocess(x):        
    '''
    Preprocessing steps:
        1. lower case
        2. expand contraction 
        3. remove puctuation
        4. lemetization
        5. removing stop words
    '''

    #lower case 
    x = str(x).lower()

    #expand contraction
    x = x.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("'ll", " will")

    #remove puctuation
    #https://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
    x=re.sub('['+string.punctuation+']',"",x)

    #lemetization of sentence
    #Lemmatisation in linguistics is the process of grouping together the inflected forms of a word so they can be analysed as a single item
    #Lemmatisation is better then stemming since it doesn't chop of the remaining part
    word_tokens = word_tokenize(x) 
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemetized_string=[lemmatizer.lemmatize(w) for w in word_tokens]
    x = ' '.join(lemetized_string)  

    #removing stop words
    stop_words=set(stopwords.words('english'))
    word_tokens = word_tokenize(x) 
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    x =' '.join(filtered_sentence)  


    return x


preprocessed_text=preprocess(selected_text)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pkl.load(f)

tokenized_text=tokenizer.texts_to_sequences([preprocessed_text])

max_length=14
tokenized_padded_text=sequence.pad_sequences(tokenized_text,maxlen=max_length,padding='post')

model = tf.keras.models.load_model('my_model.h5')

output_score=model.predict(tokenized_padded_text)

output_prob=list(output_score[0])

max_score_index=output_prob.index(max(output_prob))

st.write("Your text is related to intention: "+str(bank_classes[max_score_index])+" because it has maximum probability of "+str(output_prob[max_score_index]))

df = pd.DataFrame(list(zip(bank_classes, output_score[0])), 
               columns =['Classes', 'Score']) 

fig, ax = plt.subplots(figsize=(10,40))
ax=sns.set_theme(style="whitegrid")
ax=sns.barplot(x = 'Score',
            y = 'Classes',
            data = df)
ax.set_xlabel("Score",fontsize=30)
ax.set_ylabel("Classes",fontsize=30)

st.pyplot(fig)

#alt.Chart(df).mark_bar().encode(y="Classes",
#            x="Score",
#            tooltip=["name", "confidence"])

#fig = px.bar(df, x='Score', y='Classes')
#fig.show()

def add_reference():
    st.markdown('## Reference')
    st.markdown('* Span-ConveRT: Few-shot Span Extraction for Dialog with Pretrained Conversational Representations, https://arxiv.org/abs/2005.08866')
    st.markdown('* Efficient Intent Detection with Dual Sentence Encoders, https://arxiv.org/abs/2003.04807')
    
def contact():
    st.markdown('## About me')
    st.markdown('* Linkedin: https://www.linkedin.com/in/winston-fernandes-a14a89145/')
    st.markdown('* Email 📧: winston23fernandes.wf@gmail.com')
    st.markdown('* Contact 📱: +91-7507050685')         

add_reference()
contact()








