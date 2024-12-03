#!/usr/bin/env python
# coding: utf-8

# # Amazon Title Reviews Sentiment - Bidirectional LSTM

# In[2]:


get_ipython().system('pip install emoji')


# In[18]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("fivethirtyeight")
import seaborn as sns
sns.set_style("darkgrid")
# from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
import re, string, nltk
import emoji, bz2
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")


# In[19]:


df = pd.read_csv("./train.csv", header=None)

# Select the first and third columns
df_selected = df[[0, 2]]

# Rename the columns to 'Sentiment' and 'Review'
df_selected.columns = ['Sentiment', 'Review']

# Display the first few rows to verify
print(df_selected.head())

# Save the cleaned DataFrame with headers if needed
df_selected.to_csv('cleaned_reviews.csv', index=False)


# In[20]:


df = df_selected[["Review","Sentiment"]]
df.head()


# In[21]:


# shape of data
print(f"Data consists of {df.shape[0]} rows and {df.shape[1]} columns.")


# In[22]:


# checking for null values
df.isna().sum()


# In[23]:


# dropping null values
df = df.dropna()


# In[24]:


# checking for any duplicate in the data
df.duplicated().sum()


# In[25]:


df1 = df
df1.shape


# In[26]:


df1.Sentiment.value_counts()


# In[27]:


sns.countplot(df1.Sentiment,palette="mako")
plt.title("Countplot for Sentiment Labels")


# * Classes are balanced. So, no need for oversampling or undersampling the target column.

# # Cleaning Data

# In[28]:


def clean_text(df, field):
    df[field] = df[field].str.replace(r"@"," at ")
    df[field] = df[field].str.replace("#[^a-zA-Z0-9_]+"," ")
    df[field] = df[field].str.replace(r"[^a-zA-Z(),\"'\n_]"," ")
    df[field] = df[field].str.replace(r"http\S+","")
    df[field] = df[field].str.lower()
    return df

clean_text(df1,"Review")


# In[33]:


import nltk
nltk.download('wordnet')


# In[ ]:


# Applying Lemmmatizer to remove tenses from texts.
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    # text= re.sub(emoji.get_emoji_regexp(),"",text)
    text = [lemmatizer.lemmatize(word) for word in text.split() if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text

df1["clean_review"] = df1["Review"].apply(preprocess_text)


# In[32]:


df1.head(15)


# In[21]:


text_length = pd.Series([len(review.split()) for review in df1["clean_review"]])
text_length.plot(kind="box")
plt.ylabel("Text Length")


# In[23]:


plt.figure(figsize=(12,8))
sns.histplot(text_length,palette="deep")
plt.xlabel("Text Length")
plt.ylabel("Frequency")


# ### WordClouds
# * Useful for viewing and analyzing words that are frequently used.

# In[58]:


# Negative Review WordCloud
# plt.figure(figsize=(20,20))
# wc1 = WordCloud(max_words=2000,min_font_size=10, height=800, width=1600, 
            #    background_color="white").generate(" ".join(df[df["Sentiment"]==0].clean_review))
# plt.imshow(wc1)


# In[57]:


# Positive Review WordCloud
# plt.figure(figsize=(20,20))
# wc = WordCloud(max_words=2000,min_font_size=10, height=800, width=1600, 
#                background_color="white").generate(" ".join(df[df["sentiment"]==1].clean_review))
# # plt.imshow(wc)


# In[24]:


df = df1[["Sentiment","clean_review"]]
df.head(10)


# In[25]:


df.sentiment.unique()


# # Model Training-----------------------------------------------------------------------------------------------------

# In[26]:


X_train, X_test, y_train, y_test = train_test_split(np.array(df["clean_review"]),np.array(df["Sentiment"]), test_size=0.25,random_state=42)
print(X_train.shape)
print(X_test.shape)


# ### Term Frequency- Inverse Document Frequency (TF-IDF)

# In[27]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer(use_idf=True, tokenizer=word_tokenize)
X_train_tf2 = tfidf2.fit_transform(X_train)
X_test_tf2 = tfidf2.transform(X_test)


# #### Fitting Machine learning Models

# In[28]:


# rf = RandomForestClassifier()
# rf.fit(X_train_tf2, y_train)


# In[29]:


# from sklearn.metrics import roc_auc_score
# y_pred = rf.predict(X_test_tf2)
# acc = accuracy_score(y_pred, y_test)
# report = classification_report(y_test, y_pred)
# roc = roc_auc_score(y_test,y_pred)
# print(f"Accuracy: {acc*100}% and Roc Auc Score:{roc_auc_score(y_test,y_pred)}")
# print(report)


# In[30]:


# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# xgb = XGBClassifier(eval_metric="mlogloss")
# xgb.fit(X_train_tf2, y_train)


# In[31]:


# y_pred2 = xgb.predict(X_test_tf2)
# acc2 = accuracy_score(y_pred2, y_test)
# report2 = classification_report(y_test, y_pred2)
# roc2 = roc_auc_score(y_test,y_pred2)
# print(f"Accuracy: {acc2*100}% and Roc Auc Score:{roc_auc_score(y_test,y_pred2)}")
# print(report2)


# In[32]:


# lgb = LGBMClassifier()
# lgb.fit(X_train_tf2, y_train)


# In[33]:


# y_pred_lgb = lgb.predict(X_test_tf2)
# acc_lgb = accuracy_score(y_pred_lgb, y_test)
# report_lgb = classification_report(y_test, y_pred_lgb)
# roc_lgb = roc_auc_score(y_test,y_pred_lgb)
# print(f"Accuracy: {acc_lgb*100}% and Roc Auc Score:{roc_auc_score(y_test,y_pred_lgb)}")
# print(report_lgb)


# In[34]:


# from sklearn.naive_bayes import MultinomialNB
# nb = MultinomialNB()
# nb.fit(X_train_tf2, y_train)


# In[35]:


# y_pred3 = nb.predict(X_test_tf2)
# acc3 = accuracy_score(y_pred3, y_test)
# report3 = classification_report(y_test, y_pred3)
# roc3 = roc_auc_score(y_test,y_pred3)
# print(f"Accuracy: {acc3*100}% and Roc Auc Score:{roc_auc_score(y_test,y_pred3)}")
# print(report3)


# In[36]:


# gb = GradientBoostingClassifier()
# gb.fit(X_train_tf2, y_train)


# In[37]:


# y_pred4 = gb.predict(X_test_tf2)
# acc4 = accuracy_score(y_pred4, y_test)
# report4 = classification_report(y_test, y_pred4)
# roc4 = roc_auc_score(y_test,y_pred4)
# print(f"Accuracy: {acc4*100}% and Roc Auc Score:{roc_auc_score(y_test,y_pred4)}")
# print(report4)


# In[38]:


dt = DecisionTreeClassifier()
dt.fit(X_train_tf2, y_train)
y_pred5 = dt.predict(X_test_tf2)
acc5 = accuracy_score(y_pred5, y_test)
report5 = classification_report(y_test, y_pred5)
roc5 = roc_auc_score(y_test,y_pred5)
print(f"Accuracy: {acc5*100}% and Roc Auc Score:{roc_auc_score(y_test,y_pred5)}")
print(report5)


# In[39]:


from sklearn.metrics import precision_score
ps = precision_score(y_test, y_pred)
ps_lgb = precision_score(y_test,y_pred_lgb)
ps2 = precision_score(y_test, y_pred2)
ps3 = precision_score(y_test, y_pred3)
ps4 = precision_score(y_test, y_pred4)
ps5 = precision_score(y_pred5,y_test)


# In[40]:


accuracys = [acc,acc2,acc3,acc4,acc5,acc_lgb]
roc_scores = [roc, roc2, roc3, roc4,roc5, roc_lgb]
precision_scores = [ps,ps2,ps3,ps4,ps5,ps_lgb]
models = {"Random Forest":rf,"XGboost":xgb,"Naive Bayes":nb,"Gradient Boosting":gb,"Decision Tree":dt,"LGB Machine":lgb}

model_df = pd.DataFrame({"Models":models.keys(),"Accuracy":accuracys,"Precision Score":precision_scores,"Roc Scores": roc_scores}).sort_values("Roc Scores",ascending=False)


# In[41]:


# Summary of Machine Learning Models Performance
model_df


# # Deep Learning
# * Applying tokenizer for Bi-LSTM input after splitting data into training and testing/validation set.

# In[42]:


from sklearn.model_selection import train_test_split
X = df["clean_review"]
y = df.sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
display(X_train.shape)
display(X_test.shape)


# In[43]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)


# In[44]:


# using tokenizer to transform text messages into training and testing set
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# In[45]:


X_train_seq_padded = pad_sequences(X_train_seq, maxlen=64)
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=64)


# In[46]:


X_train_seq_padded[0]


# ### Bidirectional LSTM
# * Structure and Parameters

# In[48]:


# construct model
BATCH_SIZE = 64

from keras.utils.vis_utils import plot_model
model = Sequential()
model.add(Embedding(len(tokenizer.index_word)+1,64))
model.add(Bidirectional(LSTM(100, dropout=0,recurrent_dropout=0)))
model.add(Dense(128, activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile("adam","binary_crossentropy",metrics=["accuracy"])
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# In[49]:


# Used for preventing ovefitting
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss",patience=5,verbose=True)


# In[51]:


history = model.fit(X_train_seq_padded, y_train,batch_size=BATCH_SIZE,epochs=15,
                    validation_data=(X_test_seq_padded, y_test),callbacks=[early_stop])


# In[52]:


from sklearn.metrics import roc_auc_score
pred_train = model.predict(X_train_seq_padded)
pred_test = model.predict(X_test_seq_padded)
print('LSTM Recurrent Neural Network baseline: ' + str(roc_auc_score(y_train, pred_train)))
print('LSTM Recurrent Neural Network: ' + str(roc_auc_score(y_test, pred_test)))


# In[53]:


model.evaluate(X_test_seq_padded, y_test)


# In[54]:


acc = history.history["accuracy"]
loss = history.history["loss"]

val_acc = history.history["val_accuracy"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(9,6))
plt.plot(acc,label="Training Accuracy")
plt.plot(val_acc,label="Validation Accuracy")
plt.legend()
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")


# In[55]:


plt.figure(figsize=(9,6))
plt.plot(loss,label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.ylabel("Loss")
plt.title("Training and Validation Loss")


# * Since roc_auc_score of LSTM is best among all the models we trained, so **Bidirectional LSTM is the best model** among all the other ones which we have trained.
