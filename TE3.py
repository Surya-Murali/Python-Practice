from itertools import islice
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import pandas as pd
from collections import OrderedDict
from string import ascii_letters, punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from collections import Counter

domains_list = []
messages = []

def clean_str(string):
    string = string.replace("_", " ")
    string = string.replace("'", " ")
    string = re.sub(re.compile("<math>.*?</math>"), " ", string)
    string = re.sub(re.compile("<url>.*?</url>"), " ", string)
    string = re.sub(re.compile("<.*?>"), " ", string)
    string = re.sub(re.compile("&.*?;"), " ", string)
    string = re.sub(re.compile("/.*?>"), " ", string)
    string = re.sub(re.compile("i>"), " ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    return string

def pre_process_texts(text_file):
    text = ""
    with open("/Users/sm912r/PycharmProjects/TextFiles/" + text_file + ".txt") as fp:
        for line in islice(fp, 2, None):
            text = text + line

    text = clean_str(text)
    text = text.lower()

    words = word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    filtered_sentence = []

    for w in words:
        if w not in stop_words:
            filtered_sentence.append(w)

    # print(filtered_sentence)

    ps = PorterStemmer()

    stemmed = []

    for w in filtered_sentence:
        stemmed.append(w)

    allowed = set(ascii_letters)
    output = [word for word in stemmed if any(letter in allowed for letter in word)]

    new = ""
    for i in range(len(output)):
        new = new + " " + output[i]
    new = new.encode("ascii", errors="ignore").decode()
    result = ''.join([i for i in new if not i.isdigit()])
    # Remove words less than 3 characters long
    result = re.sub(r'\b\w{1,2}\b', '', result)
    result = ' '.join(word for word in result.split() if len(word) < 15)
    return result

# Read the list of domains you want to scrape. The CSV contains the the domains list with the header 'names'
domains_data_frame = pd.read_csv("/Users/PycharmProjects/Model.csv")

# Print the domains
# print("\nTraining data: \n", domains_data_frame)

# Print the number of domains to be scrapped
print("\nNumber of Training data: ", len(domains_data_frame), "\n")

original_text = ""

# For loop iterating the list of URLs
for i in range(0, len(domains_data_frame)):
    initial_domain = (domains_data_frame.loc[i, 'names'])
    # print(initial_domain)
    initial_category = (domains_data_frame.loc[i, 'category'])
    # print(initial_category)
    original_text = original_text + " " + pre_process_texts(initial_domain)

# print(original_text)
rare_words_list = []
counts = Counter(word_tokenize(original_text))
for i in counts:
    if counts[i]<2:
        rare_words_list.append(i)

print("Number of rare words: \n", len(rare_words_list))
    # if(counts.values() < 2):
    #     print("Less than 2", counts.keys())

# For loop iterating the list of URLs
for i in range(0, len(domains_data_frame)):
    initial_domain = (domains_data_frame.loc[i, 'names'])
    # print(initial_domain)
    initial_category = str(domains_data_frame.loc[i, 'category'])
    # print(initial_category)
    messages.append(pre_process_texts(initial_domain))

# print(messages)

vect = CountVectorizer()
#Using the fit method, our CountVectorizer() will “learn” what tokens are being used in our messages.
vect.fit(messages)

# to see what tokens have been “learned” by CountVectorizer
# print(vect.get_feature_names())

doc_term_matrix = vect.transform(messages)
repr(doc_term_matrix)

# print(doc_term_matrix) #sparse matrix

#document term matrix
#is a mathematical matrix that describes the frequency of terms that occur in a collection of documents.
#rows correspond to documents in the collection and columns correspond to terms


#n order to save space/computational power a sparse matrix is created.
#This means that only the location and value of non-zero values is saved.
#it’s advisable to keep it in sparse form especially when working with a large corpus.

df = pd.DataFrame(doc_term_matrix.toarray(), columns=vect.get_feature_names())
# print(df)

#TfidfVectorizer also creates a document term matrix from our messages.
#However, instead of filling the DTM with token counts it calculates term frequency-inverse document frequency (TF-IDF) value for each word

#The TF-IDF is the product of two weights, the term frequency and the inverse document frequency.

#"Term frequency" is a weight representing how often a word occurs in a document.
#"Inverse document" frequency is another weight representing how common a word is across documents.

#If we have several occurences of the same word in one document we can expect the TF-IDF to *increase*.
#If a word is used in many documents then the TF-IDF will *decrease*.

def createDTM(messages):
    vect = TfidfVectorizer()
    doc_term_matrix = vect.fit_transform(messages)  # create DTM

    # create pandas dataframe of DTM
    return pd.DataFrame(doc_term_matrix.toarray(), columns=vect.get_feature_names())

# 6th document list of words
# print(createDTM(messages).iloc[5])

data = createDTM(messages)
# print(data.head())

# print(data.sum(axis=0))
# print(data.head())
# print(len(data.columns))

drop_columns = rare_words_list
# print("Drop Columns list: \n", drop_columns)
# To delete the column without having to reassign df you can do:
for i in drop_columns:
    data.drop(i, axis=1, inplace=True)

data = pd.concat((data, domains_data_frame['category']), axis = 1)
print("Final Training TFIDF Data: \n", data.head())

train_features = data.iloc[:, 0:len(data.columns)-1]
train_target = data.iloc[:, len(data.columns)-1:]
train_target = train_target.astype('int')
# print(train_features)
# print(train_target)

# print("TESTINGGGGGGGGGGG")

test_domains_data_frame = pd.read_csv("/Users/PycharmProjects/Model_test.csv")
# print("\nTesting data: \n", test_domains_data_frame)

# Print the number of domains to be scrapped
print("\nNumber of Testing data:", len(test_domains_data_frame), "\n")

messages_test = []

for i in range(0, len(test_domains_data_frame)):
    initial_domain_test = (test_domains_data_frame.loc[i, 'names'])
    # print(initial_domain_test)
    initial_category_test = (test_domains_data_frame.loc[i, 'category'])
    # print(initial_category)
    messages_test.append(pre_process_texts(initial_domain_test))

vect_test = CountVectorizer()
vect_test.fit(messages_test)

doc_term_matrix_test = vect_test.transform(messages_test)
repr(doc_term_matrix_test)

df_test = pd.DataFrame(doc_term_matrix_test.toarray(), columns=vect_test.get_feature_names())

def createDTM_test(messages_test):
    vect_test = TfidfVectorizer()
    doc_term_matrix = vect_test.fit_transform(messages_test)  # create DTM

    # create pandas dataframe of DTM
    return pd.DataFrame(doc_term_matrix_test.toarray(), columns=vect_test.get_feature_names())

test_data = createDTM_test(messages_test)
test_data = pd.concat((test_data, test_domains_data_frame['category']), axis = 1)

print("Training: \n")
print(data.head())
print("Number of features of Training data: \n", len(data.columns))

# print("Testing: \n")
# print(test_data.head())
# print("Number of features of Testing data: \n", len(test_data.columns))

t3 = data[:len(test_data)]

t3 = pd.DataFrame(np.zeros((t3.shape[0], t3.shape[1])))
t3.columns = data.columns
# print(t3)

new_list = []

new_list = [item for item in test_data if item in data]

for i in new_list:
    t3[i] = test_data[i]

# print(len(t3))

test_features = t3.iloc[:,0:len(t3.columns)-1]
test_target = t3.iloc[:, len(t3.columns)-1:]

# print(test_features)
# print(test_target)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
train_features = pca.fit_transform(train_features)
test_features = pca.transform(test_features)
explained_variance = pca.explained_variance_ratio_

print("Transformed Train Features Shape: \n ", train_features.shape)
print("Transformed Test Features Shape: \n", test_features.shape)
print("Explained Variance: \n", pca.explained_variance_ratio_)
print("Explained Variance Size: \n", pca.explained_variance_ratio_.size)
# print(train_target)

logreg = LogisticRegression(random_state = 0)
logreg.fit(train_features, train_target.values.ravel())
test_pred = logreg.predict(test_features)

from sklearn.metrics import accuracy_score
# accuracy_score(test_target, y_pred)

print("Accuracy: ", str(round(100*accuracy_score(test_target, test_pred), 2)) + " %")

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(test_target, test_pred)
print("Confusion Matrix: \n", confusion_matrix)

from sklearn.metrics import classification_report
print("Classification Report: \n", classification_report(test_target, test_pred))

import matplotlib.pyplot as plt
plt.rc("font", size=14)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_target, logreg.predict(test_features))
fpr, tpr, thresholds = roc_curve(test_target, logreg.predict_proba(test_features)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# from sklearn import model_selection
# from sklearn.model_selection import cross_val_score
# kfold = model_selection.KFold(n_splits=10, random_state=None, shuffle = True)
# # modelCV = LogisticRegression()
# scoring = 'accuracy'
# results = model_selection.cross_val_score(logreg, data.iloc[:,0:len(train.columns)-1], data.iloc[:,len(train.columns)-1].values.ravel(), cv=kfold, scoring=scoring)
# # print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
# print("10-fold cross validation average accuracy: ", str(round(100*results.mean(), 2)) + " %" )

"""
test_features = test.iloc[:,0:len(test.columns)-1]
test_target = test.iloc[:, len(test.columns)-1:]
# print(test_features)
# print(test_target)

logreg = LogisticRegression()
logreg.fit(train_features, train_target.values.ravel())

test_pred = logreg.predict(test_features)

from sklearn.metrics import accuracy_score
# accuracy_score(test_target, y_pred)

print("Accuracy: ", str(round(100*accuracy_score(test_target, test_pred), 2)) + " %")

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(test_target, test_pred)
print("Confusion Matrix: \n", confusion_matrix)

from sklearn.metrics import classification_report
print("Classification Report: \n", classification_report(test_target, test_pred))

import matplotlib.pyplot as plt
plt.rc("font", size=14)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_target, logreg.predict(test_features))
fpr, tpr, thresholds = roc_curve(test_target, logreg.predict_proba(test_features)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=None, shuffle = True)
# modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(logreg, data.iloc[:,0:len(train.columns)-1], data.iloc[:,len(train.columns)-1].values.ravel(), cv=kfold, scoring=scoring)
# print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
print("10-fold cross validation average accuracy: ", str(round(100*results.mean(), 2)) + " %" ) """""
