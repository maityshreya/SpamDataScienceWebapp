import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#NLP Model
df=pd.read_csv(r"E:/DATASCIENCE_WEBAPP/spam.csv",encoding="iso-8859-1")
print(df.head())
print(df.columns)
df=df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
print(df.head())
df.rename(columns = {'v1':'labels','v2':'message'}, inplace=True)
print(df.columns)
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)
df['labels']=df['labels'].map({'ham':0, 'spam':1})
print(df.head())


def clean_data(message):
    message_without_punc=[character for character in message if character not in string.punctuation]
    message_without_punc="".join(message_without_punc)
    
    separator = ' '
    return separator.join([word for word in message_without_punc.split() if word.lower() not in stopwords.words('english') ])


df['message']=df['message'].apply(clean_data)
x=df['message']
y=df['labels']

cv=CountVectorizer()
tfid = TfidfVectorizer(max_features = 3000)
X = cv.fit_transform(x)
y = df['labels'].values
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 2)

model=MultinomialNB().fit(X_train,y_train)

predictions=model.predict(X_test)

print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test, predictions))

def predict(text):
    labels = ['Not Spam','Spam']
    X = cv.transform(text).toarray()
    p=model.predict(X)
    #return p
    s=[str(i) for i in p]
    v=int(''.join(s))
    return str('This message is looking to be: '+labels[v])
    

#print(predict(['Congratulations, you won a lottery of $2000,click the link given below.']))
    
st.title('Spam Classifier')
st.image('Images\images(2).jpeg')
user_input=st.text_input('Write you meassage or text here .')
submit=st.button('Predict Text')
if submit:
    answer = predict([user_input])
    st.text(answer)