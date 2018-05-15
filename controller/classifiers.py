import os
import pickle
from sklearn.naive_bayes import MultinomialNB

class classifiers:
    model_folders = 'uploads/models/'
    nb_classifier = MultinomialNB()
    naive_model = 'trained_naive.sav'

    def __init__(self):
        os.makedirs(self.model_folders, exist_ok=True)

    def trainNaive(self, x, y):
        self.nb_classifier.fit(x, y)
        filename = self.model_folders + self.naive_model      
        pickle.dump(self.nb_classifier, open(filename, 'wb'))

    def classifyNaive(self, x):
        pass