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
