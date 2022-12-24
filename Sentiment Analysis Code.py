#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df=pd.read_csv("../input/stock-sentiment-analysis/Stock_Dataa.csv",encoding="ISO-8859-1")


# In[ ]:


df.head(5)


# In[ ]:


print("*** Shape of dataset ***")
print()
print(df.shape)


# In[ ]:


print("*** Value Counts of Label Column ***")
print()
df['Label'].value_counts()


# In[ ]:


print("*** Null values in the dataset ***")
print()
df.isnull().sum()


# In[ ]:


print("*** Data type of columns in dataset ***")
print()
df.dtypes


# In[ ]:


print("*** Basic information about the dataset ***")
print()
df.info()


# In[ ]:


print("Starting date in dataset ==> ",df['Date'].min())

print("Ending date in dataset ==> ",df['Date'].max())


# In[ ]:


train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']


# In[ ]:


print("*** Shape of training dataset *** ", train.shape)
print("*** Shape of test dataset *** ", test.shape)


# In[ ]:


data_train=train.iloc[:,2:27]

data_train.replace("[^a-zA-Z]"," ",regex=True,inplace=True)


# In[ ]:


cols=list(data_train.columns)

for sent in cols:
    data_train[sent]=data_train[sent].str.lower()


# In[ ]:


headlines=[]

for row in range(0,len(data_train)):
    headlines.append(" ".join(str(x) for x in data_train.iloc[row,0:25]))


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from wordcloud import WordCloud
import optuna
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score , classification_report , f1_score , confusion_matrix


# In[ ]:


paragraph=" "
for paras in headlines:
    paragraph+=paras


# In[ ]:


text=paragraph

wordcloud=WordCloud(width=2000,height=1000,background_color='#F2EDD7FF',stopwords=stopwords.words('english')).generate(text)

plt.figure(figsize=(20,30))
plt.imshow(wordcloud)

plt.show()


# In[ ]:


text=headlines[0]

wordcloud=WordCloud(width=2000,height=1000,background_color='#F2EDD7FF',stopwords=stopwords.words('english')).generate(text)

plt.figure(figsize=(20,30))
plt.imshow(wordcloud)

plt.show()


# In[ ]:


cv=CountVectorizer(ngram_range=(2,2))

train_dataset=cv.fit_transform(headlines)


# In[ ]:


random_classifier= RandomForestClassifier(n_estimators=200,criterion='entropy', random_state=42)
random_classifier.fit(train_dataset,train['Label'])


# In[ ]:


test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = cv.transform(test_transform)
predictions = random_classifier.predict(test_dataset)


# In[ ]:


print("*** Accuracy score using bag of words method ***" ,round(accuracy_score(test['Label'],predictions)*100,2))


# In[ ]:


print(classification_report(test['Label'],predictions))


# In[ ]:


''data_test=test.iloc[:,2:27]

data_test.replace("[^a-zA-Z]"," ",regex=True,inplace=True)

cols=list(data_test.columns)

for sent in cols:
    data_test[sent]=data_test[sent].str.lower()
    
headlines_test=[]

for row in range(0,len(data_test)):
    headlines_test.append(" ".join(str(x) for x in data_test.iloc[row,0:25]))

    
wordnet=WordNetLemmatizer()
for head in range(len(headlines_test)):
    rev=headlines_test[head].split()
    rev1=[wordnet.lemmatize(word) for word in rev if not word in set(stopwords.words('english'))]
    rev=" ".join(rev1)
    headlines_test[head]=rev

test_dataset_wl= cv.transform(headlines_test)'''


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizerz


# In[ ]:


tf_idf=TfidfVectorizer(ngram_range=(2,2))

train_dataset_tf=tf_idf.fit_transform(headlines)


# In[ ]:


random_classifier_tf= RandomForestClassifier(n_estimators=200,criterion='entropy', random_state=42)
random_classifier_tf.fit(train_dataset_tf,train['Label'])


# In[ ]:


test_transform_tf= []
for row in range(0,len(test.index)):
    test_transform_tf.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset_tf = tf_idf.transform(test_transform)
predictions_tf = random_classifier_tf.predict(test_dataset_tf)


# In[ ]:


print("*** Accuracy using TF-IDF method ***" , round(accuracy_score(test['Label'],predictions_tf)*100,2))


# In[ ]:


print(classification_report(test['Label'],predictions_tf))


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dt_bow=DecisionTreeClassifier(random_state=42)

dt_bow.fit(train_dataset,train['Label'])


# In[ ]:


test_transform_dt= []
for row in range(0,len(test.index)):
    test_transform_dt.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset_dt = cv.transform(test_transform_dt)
predictions_dt = dt_bow.predict(test_dataset_dt)


# In[ ]:


print("*** Acuuracy score decision tree plus bag of words ***", accuracy_score(test['Label'],predictions_dt))


# In[ ]:


print("*** Classification report ***")
print()
print(classification_report(test["Label"],predictions_dt))


# In[ ]:


tf_idf_dt=TfidfVectorizer(ngram_range=(2,2))

train_dataset_tf_dt=tf_idf_dt.fit_transform(headlines)


# In[ ]:


dt_bow_tf=DecisionTreeClassifier(random_state=42)

dt_bow_tf.fit(train_dataset_tf_dt,train['Label'])


# In[ ]:


test_transform_dt_tf= []
for row in range(0,len(test.index)):
    test_transform_dt_tf.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset_dt_tf = tf_idf_dt.transform(test_transform_dt_tf)
predictions_dt_tf = dt_bow_tf.predict(test_dataset_dt_tf)


# In[ ]:


print("*** Acuuracy score decision tree plus TF-IDF ***", accuracy_score(test['Label'],predictions_dt_tf))
print()
print("*** Classification report ***")
print()
print(classification_report(test["Label"],predictions_dt_tf))


# In[ ]:


from sklearn.naive_bayes import MultinomialNB


# In[ ]:


naive = MultinomialNB()

naive.fit(train_dataset,train['Label'])


# In[ ]:


test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset_nb = cv.transform(test_transform)
predictions_nb = naive.predict(test_dataset_nb)


# In[ ]:


print("*** Accuracy score naive bayes classifier plus bag of words ***", accuracy_score(test['Label'],predictions_nb))
print()
print("*** Classification report ***")
print()
print(classification_report(test["Label"],predictions_nb))


# In[ ]:


tf_nb=TfidfVectorizer(ngram_range=(2,2))

train_dataset_nb=tf_nb.fit_transform(headlines)


# In[ ]:


naive_tf = MultinomialNB()

naive_tf.fit(train_dataset,train['Label'])


# In[ ]:


test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset_nb_tf = tf_nb.transform(test_transform)
predictions_nb_tf = naive_tf.predict(test_dataset_nb_tf)


# In[ ]:


print("*** Acuuracy score naive bayes classifier plus tf-idf ***", accuracy_score(test['Label'],predictions_nb_tf))
print()
print("*** Classification report ***")
print()
print(classification_report(test["Label"],predictions_nb_tf))


# In[ ]:


acc_score_rf1 , acc_score_rf2 = round(accuracy_score(test['Label'],predictions)*100,2) ,round(accuracy_score(test['Label'],predictions_tf)*100,2)

acc_score_dt1 , acc_score_dt2 = round(accuracy_score(test['Label'],predictions_dt)*100,2) , round(accuracy_score(test['Label'],predictions_dt_tf)*100,2)

acc_score_nb1 , acc_score_nb2 = round(accuracy_score(test['Label'],predictions_nb)*100,2) , round(accuracy_score(test['Label'],predictions_nb_tf)*100,2)
models = pd.DataFrame({
    'Model':['Random Forest (Bow)', 'Random Forest (TF-IDF)', ' Naive Bayes (BoW) ', 'Naive Bayes (TF-IDF)',"Decision Tree (BoW)" , "Decision Tree (TF-IDF)"],
    'Accuracy_score' : [acc_score_rf1, acc_score_rf2,acc_score_dt1, acc_score_dt2, acc_score_nb1,acc_score_nb2]
})


# In[ ]:


plt.figure(figsize=(20,15))
sns.barplot(y=models['Model'],x=models['Accuracy_score'],palette='cool')


# In[ ]:





# In[ ]:




