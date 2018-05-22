import os
import pandas
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from string import punctuation


class uploader:
    allowed_extensions = [".csv", ".txt", ".xls", ".xlsx"]

    def __init__(self, input_type):
        if input_type in self.allowed_extensions:
            os.makedirs("uploads/datasets/", exist_ok=True)
            os.makedirs("uploads/results/", exist_ok=True)
            self.type = input_type
            self.path = "uploads/datasets/dataset" + input_type
            self.production_path = "uploads/datasets/production" + input_type
            self.vectorizer = "uploads/datasets/vectorizer.sav"
        else:
            self.type = None
            print("Extensión invalida")

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
        dataset[data_column].apply(removePunctuation)
        print("Limpieza exitosa")
        dataset.columns = ['class_col', 'data_col']
        isbalanced = checkBalance(dataset)
        X_train, X_test, y_train, y_test = self.extractFeatures(dataset, True)
        if isbalanced == False:
            print("El dataset no está balanceado, rebalanceando...")
            X_train, y_train = balance(X_train, y_train)
            print("Balanceo realizado exitosamente")
        del isbalanced, dataset
        return X_train, y_train, X_test, y_test

    def uploadProductionDataset(self, data_column):
        if self.type == ".csv" or self.type == ".txt":
            dataset = pandas.read_csv(
                self.production_path, usecols=[data_column])
        elif self.type == ".xls" or self.type == ".xlsx":
            dataset = pandas.read_excel(
                self.production_path, names=[data_column])
        else:
            print("Tipo de dataset no válido")
            return None
        print("Se cargó correctamente el dataset, limpiando...")
        dataset = dataset[pandas.notnull(dataset[data_column])]
        dataset[data_column].apply(removePunctuation)
        print("Limpieza exitosa")
        dataset.columns = ['data_col']
        X = self.extractFeatures(dataset, False)
        return X

    def createResponse(self, y, filename, data_column):
        if self.type == ".csv" or self.type == ".txt":
            dataset = pandas.read_csv(
                self.production_path, usecols=[data_column])
        elif self.type == ".xls" or self.type == ".xlsx":
            dataset = pandas.read_excel(
                self.production_path, names=[data_column])
        else:
            print("Tipo de dataset no válido")
            return None
        print("Se cargó correctamente el dataset, limpiando...")
        dataset = dataset[pandas.notnull(dataset[data_column])]
        dataset[data_column].apply(removePunctuation)
        print("Limpieza exitosa")
        dataset['class_col'] = y
        path = "uploads/results/"
        filename += "_result.csv"
        dataset.to_csv(header=True, index=True, sep=';',
                       path_or_buf=(path + filename))

    def extractFeatures(self, dataset, split):
        stop = stopwords.words('spanish')
        stop.extend(stopwords.words('english'))
        count_vect = CountVectorizer(
            ngram_range=(1, 2), stop_words=stop, min_df=5)
        tfidf_transformer = TfidfTransformer(sublinear_tf=True)
        if split == True:
            X_train, X_test, y_train, y_test = train_test_split(
                dataset['data_col'], dataset['class_col'], random_state=0, train_size=0.7)
            # Convirtiendo a vectores de caracteristicas
            X_train_counts = count_vect.fit_transform(X_train)
            joblib.dump(count_vect, self.vectorizer)
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
            X_test_counts = count_vect.transform(X_test)
            X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
            del X_train, X_train_counts, stop, count_vect, tfidf_transformer, X_test, X_test_counts
            return X_train_tfidf, X_test_tfidf, y_train, y_test
        else:
            if os.path.exists(self.vectorizer):
                loaded_vectorizer = joblib.load(self.vectorizer, 'r')
            else:
                loaded_vectorizer = joblib.load(
                    "uploads/dataset_test/vectorizer.sav", 'r')
            X_test = dataset['data_col']
            X_test_counts = loaded_vectorizer.transform(X_test)
            del stop, count_vect, tfidf_transformer, X_test
            return X_test_counts
        return None


def removePunctuation(text):
    punctuationSigns = list(punctuation)
    punctuationSigns.extend(['¿', '¡'])
    text = ''.join([c for c in text if c not in punctuationSigns])
    del punctuationSigns
    return text


def checkBalance(dataset):
    balance = dataset.groupby('class_col').data_col.count()
    isbalanced = True
    counter = 0
    while(counter < (len(balance)-1) and isbalanced):
        if (balance[counter] != balance[counter + 1]):
            isbalanced = False
    del balance, counter
    return isbalanced


def balance(x, y):
    sampler = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = sampler.fit_sample(x, y)
    del sampler
    return X_resampled, y_resampled
