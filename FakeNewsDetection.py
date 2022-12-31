#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')


# In[7]:


print(stopwords.words('english'))


# In[8]:


news_dataset = pd.read_csv('C:\\Users\\YAMINI KHANDURI\\OneDrive\\Desktop\\train.csv')


# In[ ]:





# In[9]:


news_dataset.head()


# In[10]:


news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']


# In[11]:


print(news_dataset['content'])


# In[12]:


# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']


# In[13]:


print(X)
print(Y)


# In[14]:


from nltk.stem.porter import PorterStemmer


# In[15]:


port_stem = PorterStemmer()


# In[16]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',str(content))
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[17]:


news_dataset['content'] = news_dataset['content'].apply(stemming)


# In[18]:


print(news_dataset['content'])


# In[19]:


X= news_dataset['content'].values
Y= news_dataset['label'].values


# In[21]:


print(X)


# In[22]:


print(Y)


# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[24]:


vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[25]:


print(X)


# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)


# In[27]:


import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[28]:


from sklearn.tree import DecisionTreeClassifier
Dclassifier=DecisionTreeClassifier()
from sklearn import metrics
import numpy as np
import itertools
Dclassifier.fit(X_train,Y_train)
pred = Dclassifier.predict(X_test)
score = metrics.accuracy_score(Y_test,pred)
print("Accuracy: %0.3f"%score)
cm = metrics.confusion_matrix(Y_test,pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[29]:


Dclassifier.fit(X_train, Y_train)
pred= Dclassifier.predict(X_test)
score= metrics.accuracy_score(Y_test, pred)
score


# In[31]:


from sklearn.metrics import classification_report


# In[33]:


pred= Dclassifier.predict(X_test)
print(classification_report(Y_test, pred))


# # Random Forest Classifier

# In[34]:


from sklearn.ensemble import RandomForestClassifier
RFCclassifier=RandomForestClassifier()
from sklearn import metrics
import numpy as np
import itertools
RFCclassifier.fit(X_train,Y_train)
pred = RFCclassifier.predict(X_test)
score = metrics.accuracy_score(Y_test,pred)
print("Accuracy: %0.3f"%score)
cm = metrics.confusion_matrix(Y_test,pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[35]:


RFCclassifier.fit(X_train, Y_train)
pred= RFCclassifier.predict(X_test)
score= metrics.accuracy_score(Y_test, pred)
score


# In[31]:


pred_= RFCclassifier.predict(X_test)
print(classification_report(Y_test, pred))


# # Logistic Regression

# In[32]:


from sklearn.linear_model import LogisticRegression
model =LogisticRegression()
from sklearn import metrics
import numpy as np
import itertools
model.fit(X_train,Y_train)
pred = model.predict(X_test)
score = metrics.accuracy_score(Y_test,pred)
print("Accuracy: %0.3f"%score)
cm = metrics.confusion_matrix(Y_test,pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])


# In[33]:


model.fit(X_train, Y_train)
pred= model.predict(X_test)
score= metrics.accuracy_score(Y_test, pred)
score


# In[34]:


# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[36]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[37]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[38]:


print('Accuracy score of the test data : ', test_data_accuracy)

