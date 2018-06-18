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

domains_list = []
messages = []

def pre_process_texts(text_file):
    text = ""
    with open("/Users/PycharmProjects/TextFiles/" + text_file + ".txt") as fp:
        for line in islice(fp, 2, None):
            # print(line)
            text = text + line
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
    return result

# Read the list of domains you want to scrape. The CSV contains the the domains list with the header 'names'
domains_data_frame = pd.read_csv("/Users/sm912r/PycharmProjects/Model.csv")

# Print the domains
print("\nList of websites to be classified: \n", domains_data_frame)

# Print the number of domains to be scrapped
print("\nNumber of websites to be classified: ", len(domains_data_frame), "\n")

# For loop iterating the list of URLs
for i in range(0, len(domains_data_frame)):
    initial_domain = (domains_data_frame.loc[i, 'names'])
    # print(initial_domain)
    initial_category = (domains_data_frame.loc[i, 'category'])
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

import pandas as pd
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
data = pd.concat((data, domains_data_frame['category']), axis = 1)
print(data.head())
print(len(data.columns))

train, test = train_test_split(data, test_size=0.2)
# print(train.head())
# print(len(train))
# print(test.head())
# print(len(test))

train_features = train.iloc[:,0:len(train.columns)-1]
train_target = train.iloc[:, len(train.columns)-1:]
# print(train_features)
# print(train_target)

test_features = test.iloc[:,0:len(test.columns)-1]
test_target = test.iloc[:, len(test.columns)-1:]
# print(test_features)
# print(test_target)

logreg = LogisticRegression()
logreg.fit(train_features, train_target.values.ravel())

y_pred = logreg.predict(test_features)

from sklearn.metrics import accuracy_score
# accuracy_score(test_target, y_pred)

print(accuracy_score(test_target, y_pred))

