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
import pickle

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

train_text_list, train_doc_list, train_domains_list, train_category_df = preprocess_data("Model_LDA")
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

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(lda_model, open(filename, 'wb'))

lda_model = pickle.load(open(filename, 'rb'))

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
train_topic_predictions = lda_model[train_corpus]

def Zero_Matrix(domains_list):
    n = num_topics
    a = []
    rows = len(domains_list)
    for i in range (0, n):
        a.append(i)
    LDA_Features_df = pd.DataFrame(0, index=np.arange(rows), columns=a)
    return LDA_Features_df

def Get_LDA_Features(topic_predictions, total_features, LDA_Features_Zeros_df):
    j = 0
    for topic in topic_predictions:
        feature_list = []
        feature_values_list = []
        for i in range(0, len(topic)):
            feature_list.append(topic[i][0])
            feature_values_list.append((topic[i][1]))

        # print(feature_list)
        # print(feature_values_list)
        total_features = list(total_features)

        for i in range(0, len(feature_list)):
            LDA_Features_Zeros_df.ix[j, feature_list[i]] = feature_values_list[i]
        j+=1
    LDA_Features_df = LDA_Features_Zeros_df
    return LDA_Features_df

# j=0
# for topic in topic_predictions:
#     # print("Record ", j+1, ":")
#     # print(train_domains_list[j])
#     # print(topic, "\n")
#     feature_list = []
#     feature_values_list = []
#     for i in range(0, len(topic)):
#         feature_list.append(topic[i][0])
#         feature_values_list.append((topic[i][1]))
#
#     # print(feature_list)
#     # print(feature_values_list)
#     j += 1

# for i in range(0, len(feature_list)):
#     LDA_training_Features_df.ix[i, feature_list[i]] = feature_values_list[i]

LDA_training_Features_Zeros_df = Zero_Matrix(train_domains_list)
total_Training_features = list(LDA_training_Features_Zeros_df.columns.values)

LDA_Training_Features = Get_LDA_Features(train_topic_predictions, total_Training_features, LDA_training_Features_Zeros_df)

LDA_Training_Features = pd.concat((LDA_Training_Features, pd.DataFrame(train_category_df, columns = ["category"])), axis = 1)
print(LDA_Training_Features)
# new_data = []
# i=1
# for topic in topic_predictions:
#     # print("Record ", i, ":")
#     # print(train_domains_list[i-1])
#     # print(topic, "\n")
#     new_data.append(pd.Series(topic).values)
#     i+=1
#
# new_data = pd.DataFrame(new_data)
# print(new_data.columns.values)
# print(type(new_data.loc[0, 0]))
# print(new_data.loc[0, 1])
# print(new_data.loc[1, 0])
# print(new_data.loc[1, 1])






# Testing documents

test_text_list, test_doc_list, test_domains_list, test_category_df = preprocess_data("Test")
print("\nNumber of Testing data: ", len(test_domains_list), "\n")
# test_doc_list = preprocess_data("Test")
test_dictionary, test_corpus = get_corpus(test_doc_list)
test_topic_predictions = lda_model[test_corpus]

LDA_testing_Features_Zeros_df = Zero_Matrix(test_domains_list)
total_Testing_features = list(LDA_testing_Features_Zeros_df.columns.values)

LDA_Testing_Features = Get_LDA_Features(test_topic_predictions, total_Testing_features, LDA_testing_Features_Zeros_df)
LDA_Testing_Features = pd.concat((LDA_Testing_Features, pd.DataFrame(test_category_df, columns = ["category"])), axis = 1)

print(LDA_Testing_Features)

# i=1
# for topic in topic_predictions:
#     # print("Record ", i, ":")
#     print(test_domains_list[i-1])
#     print(topic, "\n")
#     i+=1

train_features = LDA_Testing_Features.iloc[:, 0:len(LDA_Testing_Features.columns)-1]
train_target = LDA_Testing_Features.iloc[:, len(LDA_Testing_Features.columns)-1:]

print(train_features)
print(train_target)

test_features = LDA_Testing_Features.iloc[:,0:len(LDA_Testing_Features.columns)-1]
test_target = LDA_Testing_Features.iloc[:, len(LDA_Testing_Features.columns)-1:]

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

#####
# Visualising the Training set results
# from matplotlib.colors import ListedColormap
# X_set, y_set = train_features, train_target
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#
# print(X1)
# print(X2)
#
# plt.contourf(X1, X2, logreg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.legend()
# plt.show()

#####


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
# logreg = LogisticRegression()
scoring = 'accuracy'
# results = model_selection.cross_val_score(logreg, data.iloc[:,0:len(train.columns)-1], data.iloc[:,len(train.columns)-1].values.ravel(), cv=kfold, scoring=scoring)
results = model_selection.cross_val_score(logreg, train_features, train_target.values.ravel(), cv=kfold, scoring=scoring)
# print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
print("10-fold cross validation average accuracy: ", str(round(100*results.mean(), 2)) + " %" )
