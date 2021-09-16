#!/usr/bin/env python
# coding: utf-8

# In[1]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import re
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


# ![image.png](attachment:80243c10-d2d3-4af5-be43-f2aea2d768eb.png)

# # Import Dataset

# In[41]:


train=pd.read_csv("C:\\Users\\KARANA~1\\AppData\\Local\\Temp\\Rar$DI76.032\\train.csv")
train


# In[3]:


#Checking length of each tweet
train["length"]=train["tweet"].apply(len)


# In[4]:


train.head()


# In[5]:


train.info()


# In[6]:


train.isnull().sum()


# In[7]:


#Dropping the id column as it is of no use
train=train.drop(["id"],axis=1)


# In[8]:


#Representing labels column as a countplot
import seaborn as sns
sns.countplot(train["label"])


# In[9]:


#Distributio for length of tweets
sns.distplot(train["length"])


# # Data Preprocessing

# In[10]:


pip install nltk


# In[13]:


import re
import nltk
 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# ## Number of stopwords in each tweet

# In[14]:


stop=stopwords.words("english")


# In[15]:


#Calculating the number of stopwords in a tweet
def stop_words(df):
    df['stopwords'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
    print(df[['tweet','stopwords']].head())


# In[16]:


stop_words(train)


# # Removal of Punctuation Marks

# In[18]:


#Removing the punctuations from the tweets as they do not help in prediction
def punctuation_removal(df):
    df['tweet'] = df['tweet'].str.replace('[^\w\s]','')
    print(df['tweet'].head())


# In[19]:


punctuation_removal(train)


# # Removing the most frequent words

# In[20]:


#Checking the frequency of words in all the tweets
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]
freq


# In[21]:


freq = list(freq.index)


# In[22]:


#Removing the most frequent words from the dataset
def frequent_words_removal(df):  
    df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    print(df['tweet'].head())


# In[23]:


frequent_words_removal(train)


# # Removing the rare words

# In[24]:


#Checking for rare words in the tweets
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]
freq


# In[25]:


#Removing all the rare words from the tweets
freq = list(freq.index)
def rare_words_removal(df):
    df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    print(df['tweet'].head())


# In[26]:


rare_words_removal(train)


# In[27]:


stemmer=PorterStemmer()


# # Stemming and Removing Stopwords

# In[28]:


corpus=[]
for i in range(len(train)):
    #replacing everything other than alphabets with a space
    review=re.sub("[^a-zA-Z]"," ",str(train["tweet"][i]))
    #Lowering the tweets
    review=review.lower()
    #Converting in a list
    review=review.split()
    #Finding and removing stopwords
    review=[stemmer.stem(word) for word in review if not word in set(stopwords.words("english"))]
    #Joining after removal of stopwords
    review=" ".join(review)
    corpus.append(review)


# In[29]:


corpus


# # Creating TFIDF

# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfv=TfidfVectorizer()
X=tfv.fit_transform(corpus).toarray()
y=train["label"]


# # Splitting the Dataset into Train and Test set

# In[31]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[32]:


X_train.shape,y_train.shape


# In[33]:


X_test.shape,y_test.shape


# # Training the Naive Bayes model

# In[34]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train,y_train)


# # Predicting results

# In[36]:


y_pred=classifier.predict(X_test)


# # Checking Accuracies and Creating Confusion Matrix

# In[37]:


from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_pred,y_test)


# In[38]:


acc_MNB=accuracy_score(y_pred,y_test)
acc_MNB

