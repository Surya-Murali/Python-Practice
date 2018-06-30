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

def preprocess_data(csv_file):
    # Read the list of domains you want to scrape. The CSV contains the the domains list with the header 'names'
    # domains_data_frame = pd.read_csv("/Users/sm912r/PycharmProjects/Model_LDA.csv")
    doc_list = []
    text_list = []
    domains_list = []
    category_df = []
    domains_data_frame = pd.read_csv("/Users/sm912r/PycharmProjects/" + csv_file + ".csv")
    # Print the domains
    # print("\nTraining data: \n", domains_data_frame)

    # Print the number of domains to be scrapped
    # print("\nNumber of Training data: ", len(domains_data_frame), "\n")

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
            print("Exception domain: ", initial_domain)
            domains_data_frame.drop(domains_data_frame.index[i])
            continue
        # print(len(text))
        if(len(text)<900):
            print(len(text))
            print(initial_domain)
            domains_data_frame.drop(domains_data_frame.index[i])
            continue
        domains_list.append(initial_domain)
        category_df.append(initial_category)
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
    return text_list, doc_list, domains_list, category_df

train_text_list, train_doc_list, train_domains_list, train_category_df = preprocess_data("Train")
print("\nNumber of Training data: ", len(train_domains_list), "\n")

def get_corpus(doc_list):
    # Build the bigram and trigram models
    # bigram = gensim.models.Phrases(doc_list, min_count=5, threshold=100) # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[doc_list[0]], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    # bigram_mod = gensim.models.phrases.Phraser(bigram)
    # print(bigram_mod[docs[0]])
    # trigram_mod = gensim.models.phrases.Phraser(trigram)
    # print(trigram_mod[docs[0]])

    #Remove rare & common tokens
    # Create a dictionary representation of the documents.
    from gensim.corpora.dictionary import Dictionary
    dictionary = Dictionary(doc_list)
    # dictionary.filter_extremes(no_below=10, no_above=0.2)
    #Create dictionary and corpus required for Topic Modeling
    corpus = [dictionary.doc2bow(doc) for doc in doc_list]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    return dictionary, corpus
    # print(corpus)

train_dictionary, train_corpus = get_corpus(train_doc_list)

# Set parameters.
num_topics = 15
chunksize = 500
passes = 20
iterations = 400
eval_every = 1

# Make a index to word dictionary.
temp = train_dictionary[0]  # only to "load" the dictionary.
id2word = train_dictionary.id2token
# print(id2word)
# print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
lda_model = gensim.models.ldamodel.LdaModel(corpus=train_corpus, id2word=id2word, chunksize=chunksize, alpha='auto', eta='auto',
                                            iterations=iterations, num_topics=num_topics, passes=passes, eval_every=eval_every)
# Print the Keyword in the 5 topics
print(lda_model.print_topics())

# print("############################")
# print(lda_model.get_topics())
# print(lda_model.show_topic(0))
# print("****************************")
# print(lda_model.top_topics(corpus))

top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 5)])

# print("****************************")
# print(top_words_per_topic)

# print("****************************")
# print(type(lda_model.print_topics(0)))
# print(lda_model.print_topics(0)[0])
"""
print("***June 25***")
# print(type(lda_model.print_topics(0)[0][1]))
print(lda_model.print_topics(0)[0][1])
topics = lda_model.print_topics(0)[0][1]
# print([pos for pos, char in enumerate(topics) if char == '"'])
position_list = [pos for pos, char in enumerate(topics) if char == '"']
for i in range(len(position_list)):
    if i % 2 == 0:
        s = topics[position_list[i]+1:position_list[i+1]]
        print(s)

for i in range(len(lda_model.print_topics())):
    print(i)
    print((lda_model.print_topics(0)[i]))
"""

# lda_model.save("YAYYYYYYY")

"""
import pyLDAvis.gensim
print(pyLDAvis.gensim.prepare(lda_model, corpus, dictionary))
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.show(LDAvis_prepared)

# Compute Coherence Score using c_v
coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_list, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Compute Coherence Score using UMass
coherence_model_lda = CoherenceModel(model=lda_model, texts=doc_list, dictionary=dictionary, coherence="u_mass")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=doc_list, start=2, limit=40, step=6)
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

# LDA Prediction

# Training documents

# test_doc_list = preprocess_data("Train")
# test_dictionary, test_corpus = get_corpus(test_doc_list)
topic_predictions = lda_model[train_corpus]

new_data = []
i=1
for topic in topic_predictions:
    # print("Record ", i, ":")
    # print(train_domains_list[i-1])
    # print(topic, "\n")
    new_data.append(pd.Series(topic).values)
    i+=1

new_data = pd.DataFrame(new_data)
print(new_data.columns.values)
print(type(new_data.loc[0, 0]))
print(new_data.loc[0, 1])
print(new_data.loc[1, 0])
print(new_data.loc[1, 1])

# Testing documents

test_text_list, test_doc_list, test_domains_list, test_category_df = preprocess_data("Test")
print("\nNumber of Testing data: ", len(test_domains_list), "\n")
# test_doc_list = preprocess_data("Test")
test_dictionary, test_corpus = get_corpus(test_doc_list)
topic_predictions = lda_model[test_corpus]

i=1
for topic in topic_predictions:
    # print("Record ", i, ":")
    print(test_domains_list[i-1])
    print(topic, "\n")
    i+=1
