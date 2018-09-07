import numpy as np
import pandas as pd
import re
from sklearn.metrics import mean_squared_error
from subprocess import check_output
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import nltk.data
import logging
from gensim.models import word2vec
class EntryPoint:
    def __init__(self):
        self.name = 'EntryPoint'
        self.csvReader = self.CSVReader()
        self.model = self.Model()
        self.testing = self.Testing()
        #self.submit = self.Submit()

    class CSVReader:
        static_elem = 123

        def __init__(self):
            self.train = ""
            self.test = ""

        def read_data_at(self, file_path):
            print("ls to give path" + check_output(["ls", file_path]).decode("utf8"))  # check the files available in the directory

            X = pd.read_csv(file_path, header=0, delimiter="\t", quoting=3)
            data_map = {'X': X}
            y = ""
            if 'sentiment' in X:
                y = X["sentiment"]
                del X["sentiment"]
                data_map['y'] = y
            print("train : " + str(X.shape))
            print("Header: " + str(X.columns.values))
            return data_map

        def process_sentence(self, review, remove_stopwords=False):
            # Function to convert a document to a sequence of words,
            # optionally removing stop words.  Returns a list of words.
            #
            # 1. Remove HTML
            review_text = BeautifulSoup(review).get_text()
            #
            # 2. Remove only symbols
            review_text = re.sub("[^a-zA-Z ^\d]", " ", review_text)
            #
            # 3. Convert words to lower case and split them
            words = review_text.lower().split()
            #
            # 4. Optionally remove stop words (false by default)
            if remove_stopwords:
                stops = set(stopwords.words("english"))
                words = [w for w in words if not w in stops]
            #
            # 5. Return a list of words
            return (words)

        # Define a function to split a review into parsed sentences
        def process_review_to_list_of_sentences(self, review, tokenizer, remove_stopwords=False):
            # Function to split a review into parsed sentences. Returns a
            # list of sentences, where each sentence is a list of words
            #
            # 1. Use the NLTK tokenizer to split the paragraph into sentences
            raw_sentences = tokenizer.tokenize(review.strip().decode('utf-8'))
            #
            # 2. Loop over each sentence
            sentences = []
            for raw_sentence in raw_sentences:

                # If a sentence is empty, skip it
                if len(raw_sentence) > 0:
                    # Otherwise, call review_to_wordlist to get a list of words
                    sentences.append(self.process_sentence(raw_sentence, remove_stopwords))
            #
            # Return the list of sentences (each sentence is a list of words,
            # so this returns a list of lists
            return sentences
        def process_reviews_to_vector(self, X, column_to_preprocess="review"):
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            sentences = []  # Initialize an empty list of sentences
            print("Parsing sentences from training set")
            for col_value in X[column_to_preprocess]:
                sentences += self.process_review_to_list_of_sentences(col_value, tokenizer)
            return sentences

    class Model:
        def train_word2vec_model(self, reviewColmn):
            # Import the built-in logging module and configure it so that Word2Vec
            # creates nice output messages

            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

            # Set values for various parameters
            num_features = 300  # Word vector dimensionality
            min_word_count = 40  # Minimum word count
            num_workers = 4  # Number of threads to run in parallel
            context = 10  # Context window size
            downsampling = 1e-3  # Downsample setting for frequent words

            # Initialize and train the model (this will take some time)
            print("Training model...")
            model = word2vec.Word2Vec(reviewColmn, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)

            # If you don't plan to train the model any further, calling
            # init_sims will make the model much more memory-efficient.
            model.init_sims(replace=True)

            # It can be helpful to create a meaningful model name and
            # save the model for later use. You can load it later using Word2Vec.load()
            model_name = "300features_40minwords_10context"
            model.save(model_name)
            return model
    class Testing:
        def __init__(self):
            self.n_folds = 10

        # rmsle_cv has cross_val_score method which internally calls fit and predict method of model to train
        # and make prediction, this method should use to understand behaviour of your model.
        # root mean square logrithmic error
        def rmsle_cv(self, model, X_train, y_train):
            kf = KFold(self.n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
            rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf))
            return (rmse)

        def rmsle(self, y, y_pred):
            return np.sqrt(mean_squared_error(y, y_pred))

        def split_data_set(self, X, y):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
            print("X_train matrix size" + str(X_train.shape));
            print("X_test matrix size" + str(X_test.shape))
            return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

if __name__ == '__main__':
    readerObj = EntryPoint()
    # read data from any data sources
    #training_data = readerObj.csvReader.read_data_at("./resources/testData_subset.tsv")
    training_data = readerObj.csvReader.read_data_at("./resources/UnlabeledTrainData.tsv")
    #training_data = readerObj.csvReader.read_data_at("./resources/labeledTrainData_subset.tsv")
    X_data = training_data.get("X")
    #y_data = training_data.get("y")

    #data_sets = readerObj.testing.split_data_set(X_data, y_data)


    #X_train = data_sets.get("X_train")
    X_train = readerObj.csvReader.process_reviews_to_vector(X_data, "review")
    #y_train = data_sets.get("y_train")


    # review is converted in List of List
    trained_model = readerObj.model.train_word2vec_model(X_train)

    #X_test = data_sets.get("X_test")
    #X_test = readerObj.csvReader.process_reviews_to_vector(X_test, "review")
    #y_test = data_sets.get("y_test")
    #y_expected = trained_model.predict_output_word(X_test)
    # print(readerObj.testing.rmsle(y_test, y_expected))


  #  trained_model.doesnt_match("man woman child kitchen".split()) #'kitchen'

  #  trained_model.doesnt_match("france england germany berlin".split())
    #'berlin'

  #  trained_model.doesnt_match("paris berlin london austria".split())
    #'paris'

    print(trained_model.most_similar("man"))
    # [(u'woman', 0.6056041121482849), (u'guy', 0.4935004413127899), (u'boy', 0.48933547735214233),
    #  (u'men', 0.4632953703403473), (u'person', 0.45742249488830566), (u'lady', 0.4487500488758087),
    #  (u'himself', 0.4288588762283325), (u'girl', 0.4166809320449829), (u'his', 0.3853422999382019),
    #  (u'he', 0.38293731212615967)]

    #trained_model.most_similar("queen")
    # [(u'princess', 0.519856333732605), (u'latifah', 0.47644317150115967), (u'prince', 0.45914226770401),
    #  (u'king', 0.4466976821422577), (u'elizabeth', 0.4134873151779175), (u'antoinette', 0.41033703088760376),
    #  (u'marie', 0.4061327874660492), (u'stepmother', 0.4040161967277527), (u'belle', 0.38827288150787354),
    #  (u'lovely', 0.38668593764305115)]



