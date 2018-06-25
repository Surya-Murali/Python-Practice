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
# import matplotlib.pyplot as plt
# import seaborn as sns
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from numpy import array

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

text = ""
with open("/Users/sm912r/PycharmProjects/TextFiles/paypal.com.txt") as fp:
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
text = result
words = word_tokenize(result)

######
text1 = ""
with open("/Users/sm912r/PycharmProjects/TextFiles/bankofamerica.com.txt") as fp:
    for line in islice(fp, 2, None):
        text1 = text1 + line

text1 = clean_str(text1)
text1 = text1.lower()

words = word_tokenize(text1)

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
text1 = result
words1 = word_tokenize(result)

######
print(type(words1))

# Convert to array
# docs =array([words, words1])
docs =array([words, words1])
print(docs)

# Build the bigram and trigram models
bigram = gensim.models.Phrases(docs, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[docs[0]], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
print(bigram_mod[docs[0]])
trigram_mod = gensim.models.phrases.Phraser(trigram)
print(trigram_mod[docs[0]])

#Remove rare & common tokens
# Create a dictionary representation of the documents.
from gensim.corpora.dictionary import Dictionary
dictionary = Dictionary(docs)
# dictionary.filter_extremes(no_below=10, no_above=0.2)
#Create dictionary and corpus required for Topic Modeling
corpus = [dictionary.doc2bow(doc) for doc in docs]
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))
print(corpus)

# Set parameters.
num_topics = 14
chunksize = 500
passes = 20
iterations = 400
eval_every = 1

# Make a index to word dictionary.
temp = dictionary[0]  # only to "load" the dictionary.
id2word = dictionary.id2token
print(id2word)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, alpha='auto', eta='auto',
                                            iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every)
# Print the Keyword in the 5 topics
print(lda_model.print_topics())

# Compute Coherence Score using c_v
coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Compute Coherence Score using UMass
coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence="u_mass")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs, start=2, limit=40, step=6)
# Show graph
import matplotlib.pyplot as plt
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()






"""
tok = word_tokenize(text)
tok1 = word_tokenize(text1)
# Create Dictionary
# d2word = corpora.Dictionary(data_lemmatized)
id2word = corpora.Dictionary([tok, tok1])
print(id2word)
corpus = [id2word.doc2bow(text) for text in [tok, tok1]]
print(corpus)
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])


# Build LDA model
#lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=18, random_state=100,
#                                           update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word)

print(lda_model)
print(lda_model.print_topics(num_topics=10))

# Compute Coherence Score using c_v
coherence_model_lda = CoherenceModel(model=lda_model, texts=(text, text1), dictionary=id2word, coherence='u_mass')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# # Compute Coherence Score
# coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(model=lda_model, texts=text, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ' + str(coherence_lda))

# # Compute Coherence Score
# coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(model=lda_model, texts=tok, dictionary=id2word, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print('\nCoherence Score: ' + str(coherence_lda))

"""

"""
# Create Corpus
# texts = lemmatized_texts

id2word = corpora.Dictionary(data_lemmatized)

# Term Document Frequency
corpus = [d2word.doc2bow(text) for text in d2word]
print(corpus)

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=18,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print
'\nCoherence Score: ' + str(coherence_lda)

# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
mallet_path = '~/path/to/mallet-2.0.8/bin/mallet'  # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=20, id2word=id2word)

# Compute Coherence Score
coherence_ldamalletcoherence_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word,
                                                        coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)"""
