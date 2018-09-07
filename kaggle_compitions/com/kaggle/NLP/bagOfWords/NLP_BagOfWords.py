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
class MainEntryPoint:

    def __init__(self):
        self.name = 'MainEntryPoint'
        self.csvReader = self.CSVReader()
        self.model = self.Model()
        self.testing = self.Testing()
        self.submit = self.Submit()

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

        def preprocess_data(self, data_map):
            X_all = data_map.get('X')
            print(X_all.shape)
            y_all = ""
            if 'y' in data_map:
                y_all = data_map.get("y")
                print(y_all.shape)
            # Normalization of review column
            normalized_reviews = self.normalize_column(X_all, "review")

            # Bag of Words using sklearn's CountVectorizer API
            bag_of_words_dictionary = self.bag_of_words(normalized_reviews)
            bag_of_words_dictionary["y"] = y_all
            return bag_of_words_dictionary

        def normalize_string(self, raw_review):
            # Function to convert a raw review to a string of words
            # The input is a single string (a raw movie review), and
            # the output is a single string (a preprocessed movie review)
            #
            # 1. Remove HTML
            review_text = BeautifulSoup(raw_review).get_text()
            #
            # 2. Remove non-letters
            letters_only = re.sub("[^a-zA-Z]", " ", review_text)
            #
            # 3. Convert to lower case, split into individual words
            words = letters_only.lower().split()
            #
            # 4. In Python, searching a set is much faster than searching
            #   a list, so convert the stop words to a set
            stops = set(stopwords.words("english"))
            #
            # 5. Remove stop words
            meaningful_words = [w for w in words if not w in stops]
            #
            # 6. Join the words back into one string separated by space,
            # and return the result.
            return (" ".join(meaningful_words))

        def normalize_column(self,train, column_name):
            # Initialize an empty list to hold the clean reviews
            clean_train_reviews = []
            num_reviews = train[column_name].size
            for index, row in train.iterrows():
                if ((index + 1) % 1000 == 0):
                    print("Review %d of %d\n" % (index + 1, num_reviews))
                clean_train_reviews.append(self.normalize_string(row[column_name]))
            return clean_train_reviews

        def bag_of_words(self, clean_train_reviews):
            print("Creating the bag of words...\n")
            from sklearn.feature_extraction.text import CountVectorizer
            # Initialize the "CountVectorizer" object, which is scikit-learn's
            # bag of words tool.
            vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, max_features=5000)

            # fit_transform() does two functions: First, it fits the model
            # and learns the vocabulary; second, it transforms our training data
            # into feature vectors. The input to fit_transform should be a list of
            # strings.
            review_bag_of_words = vectorizer.fit_transform(clean_train_reviews)

            # Numpy arrays are easy to work with, so convert the result to an
            # array
            return {'review_bag_of_words': review_bag_of_words.toarray(), 'vectorizer':vectorizer}

        # this method will have List of feature vector corrosponding to vocab vector.
        # Here training feature vector is 0 or 1 corosponding to vocab vector. So here
        # for visualization purpose I am calculating 1's at column vector, which inturn
        # defines how many time corrosponding vocab is found across examples.
        def print_word_count_on_feature(self, train_data_features,vocab):
            # Sum up the counts of each vocabulary word
            dist = np.sum(train_data_features, axis=0)
            # For each, printthe vocabulary word and the number of times it
            # appears in the training set
            for tag, count in zip(vocab, dist):
                print(count, tag)

    class Model:
        def forest_desicion_tree(self, X , y):
            print("Training the random forest...")
            # Initialize a Random Forest classifier with 100 trees
            forest = RandomForestClassifier(n_estimators=100)
            # Fit the forest to the training set, using the bag of words as
            # features and the sentiment labels as the response variable
            #
            # This may take a few minutes to run
            forest = forest.fit(X ,y)
            return forest

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

    class Submit:

        # submit predictions based on trained model on unlabeled test data
        def submission(self,model):
            training_data_map = readerObj.csvReader.read_data_at("./resources/unlabeledTrainData_subset.tsv")

            bag_of_words_dictionary = MainEntryPoint().csvReader.preprocess_data(training_data_map)
            X_test = bag_of_words_dictionary.get("review_bag_of_words")

            predictions = model.predict(X_test)
            # Copy the results to a pandas dataframe with an "id" column and
            # a "sentiment" column
            output = pd.DataFrame( data={"id":training_data_map.get("X")["id"], "sentiment":predictions } )

            # Use pandas to write the comma-separated output file
            output.to_csv( "./resources/Bag_of_Words_model.csv", index=False, quoting=3 )



if __name__ == '__main__':

    readerObj = MainEntryPoint()

    # read data from any data sources
    #training_data_map = readerObj.csvReader.read_data_at("./resources/labeledTrainData.tsv")
    training_data_map = readerObj.csvReader.read_data_at("./resources/labeledTrainData_subset.tsv")
    #training_data_map = readerObj.csvReader.read_data_at("./resources/labeledTrainData_smallest_subset.tsv")
    #training_data_map = readerObj.csvReader.getTrainingData_subset()

    # normalize data for NLP
    bag_of_words_dictionary = readerObj.csvReader.preprocess_data(training_data_map)

    # train model on normalized data
    X_train = bag_of_words_dictionary.get("review_bag_of_words")
    y_train = bag_of_words_dictionary.get("y")

    data_sets = readerObj.testing.split_data_set(X_train, y_train)
    X_train_subset = data_sets.get("X_train")
    y_train_subset = data_sets.get("y_train")
    print(X_train_subset.shape)
    print(y_train_subset.shape)

    X_test_subset = data_sets.get("X_test")
    y_test_subset = data_sets.get("y_test")
    print(X_test_subset.shape)
    print(y_test_subset.shape)

    model = readerObj.model.forest_desicion_tree(X_train_subset, y_train_subset)

    # Make predictions on unseen test data with trained model
    y_test_expected = model.predict(X_test_subset)
    print(readerObj.testing.rmsle(y_test_subset, y_test_expected))


    # submit predictions on seperate test data
    readerObj.submit.submission(model)













