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
import seaborn as sns
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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)

from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(train_features, train_target.values.ravel())

# Applying PCA
from sklearn.decomposition import PCA
# pca = PCA(n_components = int(round(len(domain_df)/3)))
pca = PCA(n_components = 0.85)
# pca = PCA(n_components = 5)
train_features = pca.fit_transform(train_features)
test_features = pca.transform(test_features)
explained_variance = pca.explained_variance_ratio_

print("Transformed Train Features Shape: \n ", train_features.shape)
print("Transformed Test Features Shape: \n", test_features.shape)
print("% Variance Explained: \n", str(round(100*pca.explained_variance_ratio_.sum(), 2)) + " %")
print("Number of features with which the model is fitted: \n", pca.explained_variance_ratio_.size)
# print(pca.components_)

# var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
# print(var1)
# Dump components relations with features:
# print(pd.DataFrame(pca.components_,columns=train_features,index = ['PC-1','PC-2']))
plt.semilogy(pca.explained_variance_ratio_, '--o');
plt.semilogy(pca.explained_variance_ratio_.cumsum(), '--o');

# LDA

def get_corpus(doc_list):
    from gensim.corpora.dictionary import Dictionary
    dictionary = Dictionary(doc_list)
    corpus = [dictionary.doc2bow(doc) for doc in doc_list]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    return dictionary, corpus

train_dictionary, train_corpus = get_corpus(train_doc_list)

# Set parameters.
num_topics = 7
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
filename = 'LDA_Model_Both.sav'
pickle.dump(lda_model, open(filename, 'wb'))

lda_model = pickle.load(open(filename, 'rb'))

print(lda_model.print_topics())

top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 5)])

######
import pyLDAvis.gensim
# print(pyLDAvis.gensim.prepare(lda_model, train_corpus, train_dictionary))
LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, train_corpus, train_dictionary)
# pyLDAvis.show(LDAvis_prepared)

# Compute Coherence Score using c_v
coherence_model_lda = CoherenceModel(model=lda_model, texts=train_doc_list, dictionary=train_dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# Compute Coherence Score using UMass
coherence_model_lda = CoherenceModel(model=lda_model, texts=train_doc_list, dictionary=train_dictionary, coherence="u_mass")
coherence_lda = coherence_model_lda.get_coherence()
print('\nUMass Coherence Score: ', coherence_lda)

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary=train_dictionary, corpus=train_corpus, texts=train_doc_list, start=2, limit=40, step=6)
"""
# Show graph
import matplotlib.pyplot as plt
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.draw()
# plt.show()
"""
#########

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
            LDA_Features_Zeros_df.iloc[j, feature_list[i]] = feature_values_list[i]
        j+=1
    LDA_Features_df = LDA_Features_Zeros_df
    return LDA_Features_df

LDA_training_Features_Zeros_df = Zero_Matrix(train_domains_list)
total_Training_features = list(LDA_training_Features_Zeros_df.columns.values)

LDA_Training_Features = Get_LDA_Features(train_topic_predictions, total_Training_features, LDA_training_Features_Zeros_df)

LDA_Training_Features = pd.concat((LDA_Training_Features, pd.DataFrame(train_category_list, columns = ["category"])), axis = 1)

# test_text_list, test_doc_list, test_domains_list, test_category_list = preprocess_data("Test")
# print("\nNumber of Testing data: ", len(test_domains_list), "\n")
# Personal_Finance_Testing_Records = sum(test_category_list)
# print("Number of Test Personal Finance Records: ", Personal_Finance_Testing_Records)
# print("Number of Test Non-Personal Finance Records: ", len(test_domains_list)-Personal_Finance_Testing_Records)

# test_doc_list = preprocess_data("Test")
test_dictionary, test_corpus = get_corpus(test_doc_list)
test_topic_predictions = lda_model[test_corpus]

LDA_testing_Features_Zeros_df = Zero_Matrix(test_domains_list)
total_Testing_features = list(LDA_testing_Features_Zeros_df.columns.values)

LDA_Testing_Features = Get_LDA_Features(test_topic_predictions, total_Testing_features, LDA_testing_Features_Zeros_df)
LDA_Testing_Features = pd.concat((LDA_Testing_Features, pd.DataFrame(test_category_list, columns = ["category"])), axis = 1)

LDA_Training_Features = LDA_Training_Features.iloc[:, 0:len(LDA_Training_Features.columns)-1]
LDA_Testing_Features = LDA_Testing_Features.iloc[:, 0:len(LDA_Testing_Features.columns)-1:]

train_features = pd.concat([pd.DataFrame(train_features), pd.DataFrame(LDA_Training_Features)], axis=1)
test_features = pd.concat([pd.DataFrame(test_features), pd.DataFrame(LDA_Testing_Features)], axis=1)

SVM_Classifier = svm.SVC(kernel='sigmoid', probability=True)
print(SVM_Classifier)

#Using the Training data to build the classifier
SVM_Classifier.fit(train_features, train_target.values.ravel())

#Use the classifier to predict the class of the Testing data
SVM_test_pred = SVM_Classifier.predict(test_features)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

####
print("SVM Accuracy: ", str(round(100*accuracy_score(test_target, SVM_test_pred), 2)) + " %")
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(test_target, SVM_test_pred)
print("SVM Confusion Matrix: \n", confusion_matrix)
print("SVM Classification Report: \n", classification_report(test_target, SVM_test_pred))

j = 0
for i in range(0, len(test_domains_list)):
    if(test_category_list[i] != SVM_test_pred[i]):
        print("SVM Record: ", test_domains_list[i])
        print("SVM Actual: ", test_category_list[i])
        print("SVM Predicted: ", SVM_test_pred[i])
        j+=1

print("SVM Total Mismatches: ", j)

# import matplotlib.pyplot as plt
plt.rc("font", size=14)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
SVM_roc_auc = roc_auc_score(test_target, SVM_Classifier.predict(test_features))
SVMfpr, SVMtpr, SVMthresholds = roc_curve(test_target, SVM_Classifier.predict_proba(test_features)[:,1])
plt.figure()
plt.plot(SVMfpr, SVMtpr, label='SVM (area = %0.2f)' % SVM_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('SVM_ROC')
plt.draw()

plt.show()

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=None, shuffle = True)
scoring = 'accuracy'
results = model_selection.cross_val_score(SVM_Classifier, train_features, train_target.values.ravel(), cv=kfold, scoring=scoring)
print("SVM 10-fold cross validation average accuracy: ", str(round(100*results.mean(), 2)) + " %" )

# pyLDAvis.show(LDAvis_prepared)
