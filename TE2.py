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
