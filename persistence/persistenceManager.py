import matplotlib.pyplot as pyplot
import nltk
import numpy
import pandas
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from string import punctuation


class uploader:
    allowed_extensions = [".csv", ".txt", ".xls", ".xlsx"]

    def __init__(self, type):
        if type in self.allowed_extensions:
            self.type = type
            self.path = "uploads/datasets/dataset" + type
        else:
            self.type = None
            print("Extensión invalida")

    def checkBalance(self, dataset):
        balance = dataset.groupby('class_col').data_col.count()
        isbalanced = True
        counter = 0
        while(counter < (len(balance)-1) and isbalanced):
            if (balance[counter] != balance[counter + 1]):
                isbalanced = False
        del balance, counter
        return isbalanced

    def removePunctuation(self, text):
        punctuationSigns = list(punctuation)
        punctuationSigns.extend(['¿', '¡'])
        text = ''.join([c for c in text if c not in punctuationSigns])
        del punctuationSigns
        return text

    def extractFeatures(self, dataset):
        stop = stopwords.words('spanish')
        stop.extend(stopwords.words('english'))
        count_vect = CountVectorizer(
            ngram_range=(1, 2), stop_words=stop, min_df=5)
        tfidf_transformer = TfidfTransformer(sublinear_tf=True)
        X_train, X_test, y_train, y_test = train_test_split(
            dataset['data_col'], dataset['class_col'], random_state=0, train_size=0.7)
        # Convirtiendo a vectores de caracteristicas
        X_train_counts = count_vect.fit_transform(X_train)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        X_test_counts = count_vect.transform(X_test)
        X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
        del X_train, X_train_counts, stop, count_vect, tfidf_transformer, X_test, X_test_counts
        return X_train_tfidf, X_test_tfidf, y_train, y_test

    def balance(self, x, y):
        sampler = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = sampler.fit_sample(x, y)
        del sampler
        return X_resampled, y_resampled

    def uploadDataset(self, data_column, class_column):
        if self.type == ".csv" or self.type == ".txt":
            dataset = pandas.read_csv(
                self.path, usecols=[class_column, data_column])
        elif self.type == ".xls" or self.type == ".xlsx":
            dataset = pandas.read_excel(
                self.path, names=[class_column, data_column])
        else:
            print("Tipo de dataset no válido")
            return None
        print("Se cargó correctamente el dataset, limpiando...")
        dataset = dataset[pandas.notnull(dataset[data_column])]
        dataset[data_column].apply(self.removePunctuation)
        print("Limpieza exitosa")
        dataset.columns = ['class_col', 'data_col']
        isbalanced = self.checkBalance(dataset)
        X_train, X_test, y_train, y_test = self.extractFeatures(dataset)
        if isbalanced == False:
            print("El dataset no está balanceado, rebalanceando...")
            X_train, y_train = self.balance(X_train, y_train)
            print("Balanceo realizado exitosamente")
        del isbalanced, dataset
        return X_train, y_train, X_test, y_test
