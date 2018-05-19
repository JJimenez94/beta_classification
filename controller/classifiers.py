import os
import pickle
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB

class classifiers:
    model_folders = 'uploads/models/'
    nb_classifier = BernoulliNB()
    naive_model = 'trained_naive.sav'

    def __init__(self):
        os.makedirs(self.model_folders, exist_ok=True)

    def testNaive(self, x, y):
        expected = y
        predicted = self.nb_classifier.predict(x)
        # Se muestran las metricas resultantes del entrenamiento
        print(metrics.classification_report(expected, predicted))
        print(metrics.confusion_matrix(expected, predicted))

    def trainNaive(self, x_train, y_train, x_test, y_test, user_alpha=1.0, user_binarize=0.0):
        if user_alpha != 1.0 or  user_binarize != 0.0:
            print('Redefiniendo al clasificador Bayesiano con parametros ingresados')
            self.nb_classifier = BernoulliNB(alpha=user_alpha, binarize=user_binarize)
        self.nb_classifier.fit(x_train, y_train)
        print(self.nb_classifier)
        filename = self.model_folders + self.naive_model
        pickle.dump(self.nb_classifier, open(filename, 'wb'))
        self.testNaive(x_test, y_test)

    def classifyNaive(self, x):
        filename = self.model_folders + self.naive_model
        if os.path.exists(filename):
            loaded_naive = pickle.load(open(filename, 'rb'))
            return loaded_naive.predict(x)
        else:
            print('El modelo no ha sido entrenado previamente, omitiendo clasificación')
            return None