import os
import numpy as np
import re
import time
import string
from sklearn import metrics
import xml.etree.ElementTree as ET
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from unicodedata import normalize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from spellchecker import SpellChecker
spell_de = SpellChecker(language= 'de')
spell_fr = SpellChecker(language= 'fr')
spell_en = SpellChecker(language= 'en')
path = 'data'
language_set = {'english','french','german'}


## Vectorizing and Creating TF-IDF weighted word Embedding
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df = 0.05)
        tfidf.fit(X)
        print ('Vocab Size' , len(tfidf.vocabulary_))
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self,tfidf.vocabulary_.items()

    def transform(self, X):
        np_ar = np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
        return np_ar
    
## Vectorizing and Creating Count and binary weighted word Embedding
class CountEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X):
        count_vect = CountVectorizer(analyzer=lambda x: x, min_df = 0.0005)
        count_vect.fit(X)
        print ('Vocab Size' , len(count_vect.vocabulary_))
        return self,count_vect.vocabulary_.items()

    def transform(self, X):
        np_ar = np.array([
                np.mean([self.word2vec[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
        return np_ar
		
## Vectorizing and Creating TF-IDF weighted average word Embedding		
class TfidfWeightedAvgEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df = 0.001)
        tfidf.fit(X)
        print ('Vocab Size' , len(tfidf.vocabulary_))
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self,tfidf.vocabulary_.items()

    def transform(self, X):
        doc2vec = []
        for words in X:
            vec2sum = []
            den_val = 0
            for w in words:
                if w in self.word2vec:
                    weighted_word = self.word2vec[w]* self.word2weight[w]
                    den_val += self.word2weight[w]
                else:
                    weighted_word = np.zeros(self.dim)
                vec2sum.append(weighted_word)
            if den_val != 0:
                doc2vec.append(np.sum(vec2sum,axis=0)/den_val)
            else:
                doc2vec.append(np.sum(vec2sum,axis=0))
        return np.array(doc2vec)
		
def getvalueofnode(node):
	return node.text if node is not None else None

def read_w2v(language):
	'''
	Reading Word Embeddings 
	'''
	if(language == 'french'):
		file_name = 'wiki.multi.fr.vec'
	elif(language == 'english'):
		file_name = 'wiki.multi.en.vec'
	elif(language == 'german'):
		file_name = 'wiki.multi.de.vec'
	else:
		return False
	w2v = {}
	count = 0
	print('Loading word vectors for',language) 
	with open(os.path.join(path,file_name), "r", encoding="utf8") as lines:
		for line in lines:
			lineArr = line.split()
			x = []
			for value in lineArr[len(lineArr)-300:]:
					x.append(float(value))
			w2v[' '.join(lineArr[0:len(lineArr)-300])]=  np.array(x)
	return w2v

def tokenize(doc, keep_punctuation=True):
	'''
	Tokenizing a document
	'''
	doc = doc.lower()
	if(keep_punctuation==False):
		return np.array(re.findall("[\w]+", doc))
	else:
		token_list = doc.split()
		tokenized_list=[]
		for token in token_list:
			new_token=''
			for i in range(0,len(token)):
				if(token[i] not in string.punctuation):
					new_token+=token[i]
			tokenized_list.append(new_token)
	return tokenized_list

def cleaningTextTokenizing(line, lang = 'english'):
	'''
	Cleaning a document 
	'''
	if (lang == 'english'):
		re_print = re.compile('[^%s]' % re.escape(string.printable))
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		line = tokenize(line)
		line = [re_print.sub('', w) for w in line]
		line = [word for word in line if word.isalpha()]
	else:
		line = tokenize(line, keep_punctuation=False)
		line = [word for word in line if word.isalpha()]
	return line

# Stop word removal of tokenized input data
def get_lemmatized_text(tokenized_review):
	lemmatizer = WordNetLemmatizer()
	lemmatizedStr = []
	for word in tokenized_review:
		lemmatizedStr.append(lemmatizer.lemmatize(word))
	return lemmatizedStr

def correct_spelling(spell, misspelled):
	'''
	Returning the corrected spelling by using SpellChecker
	'''
    correctword = []
    for word in misspelled:
        corword = spell.correction(word)
        s2 = re.sub(r'[^\w\s]',' ',corword)
        for x in s2.split():
            correctword.append(x)
    return correctword

def data_parse(parsed_xml_data, language):
	'''
	Parsing the data XML format, Preprocessing it and storing the tokenize reviews along with its True Labels.
	'''
	X_data = []
	y_data = []
	for node in parsed_xml_data.getroot():
		
		summary = node.find('summary')
		rating = node.find('rating')
		text = node.find('text')
		tokens_summary = getvalueofnode(summary)
		tokens_text = getvalueofnode(text)
		if(tokens_summary == None and tokens_text == None ):
			tokens = []
		elif(tokens_text == None):
			tokens = cleaningTextTokenizing(tokens_summary,language)
		elif(tokens_summary == None):
			tokens = cleaningTextTokenizing(tokens_text,language)
		else:
			tokens = [*cleaningTextTokenizing(tokens_summary,language), *cleaningTextTokenizing(tokens_text,language)]
		
		lemmaStr = get_lemmatized_text(tokens)
		X_data.append(lemmaStr)
		if(float(getvalueofnode(rating))>3):
			y_data.append('positive')
		else:
			y_data.append('negative')
		
	return X_data, y_data

def load_data(language):
	'''
	Reading the Train and test data set
	'''
	if language in language_set:
			x_path = os.path.join(path,'amazon-dataset',language)
			print(x_path)
			parsed_xml_train = ET.parse(os.path.join(x_path+'/books/train.review'))
			parsed_xml_test = ET.parse(os.path.join(x_path+'/books/test.review'))
			X_train, y_train = data_parse(parsed_xml_train,language)
			X_test, y_test = data_parse(parsed_xml_test,language)
			return X_train, y_train, X_test, y_test

	else:
			print('%s not available' %language)
			return False

	
def words_not_w2vec(vocab, w2v):
	'''
	Extracting corpus words not present in word embeddings and its percent
	'''
	new_words = []
	percent_not_words = 0
	for word,i in list(vocab):
		if(word not in w2v):
			new_words.append(word)
	percent_not_words = (len(new_words) / len(vocab)) * 100
	return new_words, percent_not_words

def vectorize_train(w2v_lang, X_data, y_data, language):
	'''
	Training the source language train dataset 
	'''
	t0 = time.time()
	
	print('------------------------------------------------------------')
	print('Training model on',language)
	vectorizer = TfidfEmbeddingVectorizer(w2v_lang)
	vect,vocab = vectorizer.fit(X_data)
	unknown_words_list_en, percentage = words_not_w2vec(vocab,w2v_lang)
	X_t = vectorizer.transform(X_data)
	clfLSVC = LinearSVC()
	LSVC = CalibratedClassifierCV(clfLSVC)
	LSVC.fit(X_t ,y_data)
	print('Number of words not in word2vec for training:',len(unknown_words_list_en))
	print('Time taken to train the model: %f seconds' %(time.time()-t0))
	return LSVC
	
def vectorize_predict(w2v_lang, X_data, y_data, clf, language):
	'''
	testing the target language dataset 
	'''
	t0 = time.time()
	print('------------------------------------------------------------')
	print('Testing model on',language)
	vectorizer_data = TfidfEmbeddingVectorizer(w2v_lang)
	X ,vocab = vectorizer_data.fit(X_data)
	X_vect_data = vectorizer_data.transform(X_data)
	result = clf.predict(X_vect_data)
	print ('Accuracy Score ', metrics.accuracy_score(y_data, result))
	print ('Confusion matrix \n',metrics.confusion_matrix(y_data, result))
	unknown_words_list, percentage = words_not_w2vec(vocab, w2v_lang)
	print('Number of words not in word2vec for testing:',len(unknown_words_list))

def main():
	w2v_en = read_w2v('english')
	w2v_fr = read_w2v('french')
	w2v_de = read_w2v('german')
	
	#English
	X_train_en, y_train_en, X_test_en, y_test_en = load_data('english')

	#French
	X_train_fr, y_train_fr, X_test_fr, y_test_fr = load_data('french')

	#German
	X_train_de, y_train_de, X_test_de, y_test_de = load_data('german')

	### En-En, En-Fr, En-De

	clf_en = vectorize_train(w2v_en, X_train_en, y_train_en,'english')
	vectorize_predict(w2v_en, X_test_en,y_test_en, clf_en, 'english')
	vectorize_predict(w2v_fr, X_test_fr,y_test_fr, clf_en, 'french')
	vectorize_predict(w2v_de, X_test_de,y_test_de, clf_en,'german')

	### Fr-Fr, Fr-En, Fr-De
	clf_fr = vectorize_train(w2v_fr, X_train_fr, y_train_fr,'french')
	vectorize_predict(w2v_fr, X_test_fr,y_test_fr, clf_en, 'french')
	vectorize_predict(w2v_en, X_test_en,y_test_en, clf_fr, 'english')
	vectorize_predict(w2v_de, X_test_de,y_test_de, clf_fr,'german')

	### De-De, De-En, De-Fr
	clf_de = vectorize_train(w2v_de, X_train_de, y_train_de,'german')
	vectorize_predict(w2v_de, X_test_de,y_test_de, clf_de,'german')
	vectorize_predict(w2v_en, X_test_en,y_test_en, clf_de, 'english')
	vectorize_predict(w2v_fr, X_test_fr, y_test_fr, clf_de,'french')
	
	
if __name__ == '__main__':
  main()