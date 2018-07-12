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
import matplotlib.pyplot as plt
# import seaborn as sns
import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from numpy import array
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

domains_list = []
messages = []

def clean_str(string):
    string = string.replace("_", " ")
    string = string.replace("'", " ")
    string = string.replace(",", " ")
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

def preprocess_data(csv_file):
    # Read the list of domains you want to scrape. The CSV contains the the domains list with the header 'names'
    # domains_data_frame = pd.read_csv("/Users/sm912r/PycharmProjects/Model_LDA.csv")
    doc_list = []
    text_list = []
    domains_list = []
    category_list = []
    domains_data_frame = pd.read_csv("/Users/sm912r/PycharmProjects/" + csv_file + ".csv")

    # For loop iterating the list of URLs
    for i in range(0, len(domains_data_frame)):
        initial_domain = (domains_data_frame.loc[i, 'names'])
        # print(initial_domain)
        initial_category = (domains_data_frame.loc[i, 'category'])
        text = ""
        try:
            with open("/Users/sm912r/PycharmProjects/TextFiles_Jun28/"+initial_domain+".txt") as fp:
                for line in islice(fp, 2, None):
                    text = text + line
        except:
            # print("Exception domain: ", initial_domain)
            domains_data_frame.drop(domains_data_frame.index[i])
            continue
        # print(len(text))
        if(len(text)<900):
            # print(len(text))
            # print(initial_domain)
            domains_data_frame.drop(domains_data_frame.index[i])
            continue
        domains_list.append(initial_domain)
        category_list.append(initial_category)
        text = clean_str(text)
        text = text.lower()
        words = word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        filtered_sentence = []
        for w in words:
            if w not in stop_words:
                filtered_sentence.append(w)
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
        text_list.append(result)
        # Tokenizinggggggggg
        words_tok = word_tokenize(result)
        doc_list.append(words_tok)
    return text_list, doc_list, domains_list, category_list

train_text_list, train_doc_list, train_domains_list, train_category_list = preprocess_data("Train")

print("\nNumber of Training data: ", len(train_domains_list), "\n")
Personal_Finance_Train_Records = sum(train_category_list)
print("Number of Personal Finance Training Records: ", Personal_Finance_Train_Records)
print("Number of Non-Personal Finance Training Records: ", len(train_domains_list)-Personal_Finance_Train_Records)

original_text = ""
for i in range(0, len(train_text_list)):
    original_text = original_text + " " + train_text_list[i]

rare_words_list = []
counts = Counter(word_tokenize(original_text))
for i in counts:
    if counts[i]<1:
        rare_words_list.append(i)

print("Number of rare words: \n", len(rare_words_list))

vect = CountVectorizer()
#Using the fit method, our CountVectorizer() will “learn” what tokens are being used in our messages.
# vect.fit(messages)
vect.fit(train_text_list)

# to see what tokens have been “learned” by CountVectorizer
# print(vect.get_feature_names())

doc_term_matrix = vect.transform(train_text_list)
repr(doc_term_matrix)
# print(doc_term_matrix) #sparse matrix

#document term matrix
#is a mathematical matrix that describes the frequency of terms that occur in a collection of documents.
#rows correspond to documents in the collection and columns correspond to terms


#In order to save space/computational power a sparse matrix is crea sted.
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

data = createDTM(train_text_list)

drop_columns = rare_words_list
# print("Drop Columns list: \n", drop_columns)
# To delete the column without having to reassign df you can do:
for i in drop_columns:
    data.drop(i, axis=1, inplace=True)

# data = pd.concat((data, domains_data_frame['category']), axis = 1)
data = pd.concat((data, pd.DataFrame(train_category_list, columns = ["category"])), axis = 1)
print("Training TFIDF Data: \n", data.head())
print("Training TFIDF Data: \n", data.tail())
# print(pd.DataFrame(train_category_list, columns = ["category"]))

train_features = data.iloc[:, 0:len(data.columns)-1]
train_target = data.iloc[:, len(data.columns)-1:]

# print("TESTINGGGGGGGGGGG")

test_text_list, test_doc_list, test_domains_list, test_category_list = preprocess_data("Test")

print("\nNumber of Testing data: ", len(test_domains_list), "\n")
Personal_Finance_Testing_Records = sum(test_category_list)
print("Number of Test Personal Finance Records: ", Personal_Finance_Testing_Records)
print("Number of Test Non-Personal Finance Records: ", len(test_domains_list)-Personal_Finance_Testing_Records)

vect_test = CountVectorizer()
vect_test.fit(test_text_list)

doc_term_matrix_test = vect_test.transform(test_text_list)
repr(doc_term_matrix_test)

df_test = pd.DataFrame(doc_term_matrix_test.toarray(), columns=vect_test.get_feature_names())

def createDTM_test(messages_test):
    vect_test = TfidfVectorizer()
    doc_term_matrix = vect_test.fit_transform(messages_test)  # create DTM
    # create pandas dataframe of DTM
    return pd.DataFrame(doc_term_matrix_test.toarray(), columns=vect_test.get_feature_names())

test_data = createDTM_test(test_text_list)
test_data = pd.concat((test_data, pd.DataFrame(test_category_list, columns = ["category"])), axis = 1)

t3 = data[:len(test_data)]
t3 = pd.DataFrame(np.zeros((t3.shape[0], t3.shape[1])))
t3.columns = data.columns

new_list = []
new_list = [item for item in test_data if item in data]

for i in new_list:
    t3[i] = test_data[i]

test_features = t3.iloc[:,0:len(t3.columns)-1]
test_target = t3.iloc[:, len(t3.columns)-1:]

logreg = LogisticRegression()
logreg.fit(train_features, train_target.values.ravel())
fi = logreg.coef_

m = max(fi[0])

fi_list = []
for i in range (0, len(fi[0])):
    fi_list.append(fi[0][i])

X = []
Y = []

fi_pos = np.argsort(fi_list)[::-1][:15]
for i in range (0, len(fi_pos)):
    print(train_features.columns.values[fi_pos[i]])
    X.append(train_features.columns.values[fi_pos[i]])

for i in range (0, len(fi_pos)):
    print(fi_list[fi_pos[i]])
    Y.append(fi_list[fi_pos[i]])

# print(sorted(fi_list, reverse = True))

# save the model to disk
filename = 'LR_model.sav'
pickle.dump(logreg, open(filename, 'wb'))
logreg = pickle.load(open(filename, 'rb'))
test_pred = logreg.predict(test_features)

from sklearn.metrics import accuracy_score
print("LR Accuracy: ", str(round(100*accuracy_score(test_target, test_pred), 2)) + " %")

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(test_target, test_pred)
print("LR Confusion Matrix: \n", confusion_matrix)

from sklearn.metrics import classification_report
print("LR Classification Report: \n", classification_report(test_target, test_pred))

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('msurya25', 'FshKXKWREEFdeOaPuAuL')
trace1 = {
  "x": X,
  "y": Y,
  "type": "bar"
}
data = Data([trace1])
fig = Figure(data=data)
plot_url = py.plot(fig)
