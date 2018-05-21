import itertools
import matplotlib
matplotlib.use('Agg')
import numpy
import os
from matplotlib import pyplot
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


class classifiers:
    model_folders = 'uploads/models/'
    nb_classifier = BernoulliNB()
    naive_model = 'trained_naive.sav'
    nb_parameters = {'alpha': list(numpy.arange(0, 1.1, 0.01)), 'binarize': list(
        numpy.arange(0, 1.1, 0.01)), 'fit_prior': [True, False]}
    dt_classifier = DecisionTreeClassifier()
    dt_model = 'trained_dt.sav'
    dt_parameters = {'criterion': ['gini', 'entropy']}
    svm_classifier = LinearSVC()
    svm_model = 'trained_svm.sav'
    svm_parameters = {'loss': ['hinge', 'squared_hinge'],
                      'C': list(numpy.arange(0.1, 1.1, 0.01))}
    knn_classifier = KNeighborsClassifier(algorithm='auto', n_neighbors=20)
    knn_model = 'trained_knn.sav'
    ann_classifier = MLPClassifier()
    ann_model = 'trained_ann.sav'

    def __init__(self):
        os.makedirs(self.model_folders, exist_ok=True)

    def testNaive(self, x, expected):
        labels = expected.drop_duplicates()
        predicted = self.nb_classifier.predict(x)
        # Se muestran las metricas resultantes del entrenamiento
        result = metrics.classification_report(expected, predicted)
        result = report_format(result)
        save_confusion_matrix(metrics.confusion_matrix(
            expected, predicted), labels, "static/naive.png", False)
        filename = self.model_folders + self.naive_model
        elements = ["<h3> Reporte de resultados para el clasificador ingenuo de Bayes </h3> <br>", result, "<br>", "<center> <img src='static/naive.png' alt='Matriz de confusión clasificador ingenuo de Bayes'>",
                    "</center> <br> Para descargar el modelo clasificado haga click: ", "<a href=" + filename + ">En este enlace</a>"]
        del labels
        return elements

    def trainNaive(self, x_train, y_train, x_test, y_test):
        print(self.nb_classifier)
        print('Redefiniendo el clasificador Bayesiano')
        self.nb_classifier = RandomizedSearchCV(
            estimator=self.nb_classifier, param_distributions=self.nb_parameters, n_iter=100)
        print(self.nb_classifier)
        self.nb_classifier.fit(x_train, y_train)
        filename = self.model_folders + self.naive_model
        joblib.dump(self.nb_classifier, filename)
        return self.testNaive(x_test, y_test)

    def classifyNaive(self, x):
        filename = self.model_folders + self.naive_model
        if os.path.exists(filename):
            loaded_naive = joblib.load(filename, 'rb')
            return loaded_naive.predict(x)
        else:
            print('El modelo no ha sido entrenado previamente, omitiendo clasificación')
            return None

    def testDT(self, x, expected):
        labels = expected.drop_duplicates()
        predicted = self.dt_classifier.predict(x)
        # Se muestran las metricas resultantes del entrenamiento
        result = metrics.classification_report(expected, predicted)
        result = report_format(result)
        save_confusion_matrix(metrics.confusion_matrix(
            expected, predicted), labels, "static/dt.png", False)
        filename = self.model_folders + self.dt_model
        elements = ["<h3> Reporte de resultados para el árbol de decisión </h3> <br>", result, "<br>", "<center>", "<img src='static/dt.png' alt='Matriz de confusión árbol de decisión'>",
                    "</center> <br> Para descargar el modelo clasificado haga click: ", "<a href=" + filename + ">En este enlace</a>"]
        del labels
        return elements

    def trainDT(self, x_train, y_train, x_test, y_test):
        print(self.dt_classifier)
        print('Redefiniendo el clasificador Arbol de decisión')
        self.dt_classifier = RandomizedSearchCV(
            estimator=self.dt_classifier, param_distributions=self.dt_parameters, n_iter=2)
        print(self.dt_classifier)
        self.dt_classifier.fit(x_train, y_train)
        filename = self.model_folders + self.dt_model
        joblib.dump(self.dt_classifier, filename)
        return self.testDT(x_test, y_test)

    def classifyDT(self, x):
        filename = self.model_folders + self.dt_model
        if os.path.exists(filename):
            loaded_dt = joblib.load(filename, 'rb')
            return loaded_dt.predict(x)
        else:
            print('El modelo no ha sido entrenado previamente, omitiendo clasificación')
            return None

    def testSVM(self, x, expected):
        labels = expected.drop_duplicates()
        predicted = self.svm_classifier.predict(x)
        # Se muestran las metricas resultantes del entrenamiento
        result = metrics.classification_report(expected, predicted)
        result = report_format(result)
        save_confusion_matrix(metrics.confusion_matrix(
            expected, predicted), labels, "static/svm.png", False)
        filename = self.model_folders + self.svm_model
        elements = ["<h3> Reporte de resultados para el clasificador máquinas de soporte vectorial </h3> <br>", result, "<br>", "<center>", "<img src='static/svm.png' alt='Matriz de confusión SVM '>",
                    "</center> <br>", "Para descargar el modelo clasificado haga click: ", "<a href=" + filename + ">En este enlace</a>"]
        del labels
        return elements

    def trainSVM(self, x_train, y_train, x_test, y_test):
        print(self.svm_classifier)
        print('Redefiniendo el clasificador SVM')
        self.svm_classifier = RandomizedSearchCV(
            estimator=self.svm_classifier, param_distributions=self.svm_parameters, n_iter=100)
        print(self.svm_classifier)
        self.svm_classifier.fit(x_train, y_train)
        filename = self.model_folders + self.svm_model
        joblib.dump(self.svm_classifier, filename)
        return self.testSVM(x_test, y_test)

    def classifySVM(self, x):
        filename = self.model_folders + self.svm_model
        if os.path.exists(filename):
            loaded_svm = joblib.load(filename, 'rb')
            return loaded_svm.predict(x)
        else:
            print('El modelo no ha sido entrenado previamente, omitiendo clasificación')
            return None

    def testKNN(self, x, expected):
        labels = expected.drop_duplicates()
        predicted = self.knn_classifier.predict(x)
        # Se muestran las metricas resultantes del entrenamiento
        result = metrics.classification_report(expected, predicted)
        result = report_format(result)
        save_confusion_matrix(metrics.confusion_matrix(
            expected, predicted), labels, "static/knn.png", False)
        filename = self.model_folders + self.knn_model
        elements = ["<h3> Reporte de resultados para el clasificador K-Nearest Neighbors </h3> <br>", result, "<br>", "<center>", "<img src='static/knn.png' alt='Matriz de confusión KNN '>",
                    "</center> <br>", "Para descargar el modelo clasificado haga click: ", "<a href=" + filename + ">En este enlace</a>"]
        del labels
        return elements

    def trainKNN(self, x_train, y_train, x_test, y_test):
        print(self.knn_classifier)
        self.knn_classifier.fit(x_train, y_train)
        filename = self.model_folders + self.knn_model
        joblib.dump(self.knn_classifier, filename)
        return self.testKNN(x_test, y_test)

    def classifyKNN(self, x):
        filename = self.model_folders + self.knn_model
        if os.path.exists(filename):
            loaded_knn = joblib.load(filename, 'rb')
            return loaded_knn.predict(x)
        else:
            print('El modelo no ha sido entrenado previamente, omitiendo clasificación')
            return None

    def testANN(self, x, expected):
        labels = expected.drop_duplicates()
        predicted = self.ann_classifier.predict(x)
        # Se muestran las metricas resultantes del entrenamiento
        result = metrics.classification_report(expected, predicted)
        result = report_format(result)
        save_confusion_matrix(metrics.confusion_matrix(
            expected, predicted), labels, "static/ann.png", False)
        filename = self.model_folders + self.ann_model
        elements = ["<h3> Reporte de resultados para el clasificador Red Neuronal Artificial </h3> <br>", result, "<br>", "<center>", "<img src='static/ann.png' alt='Matriz de confusión ANN '>",
                    "</center> <br>", "Para descargar el modelo clasificado haga click: ", "<a href=" + filename + ">En este enlace</a>"]
        del labels
        return elements

    def trainANN(self, x_train, y_train, x_test, y_test):
        print(self.ann_classifier)
        self.ann_classifier.fit(x_train, y_train)
        filename = self.model_folders + self.ann_model
        joblib.dump(self.ann_classifier, filename)
        return self.testANN(x_test, y_test)

    def classifyANN(self, x):
        filename = self.model_folders + self.ann_model
        if os.path.exists(filename):
            loaded_ann = joblib.load(filename, 'rb')
            return loaded_ann.predict(x)
        else:
            print('El modelo no ha sido entrenado previamente, omitiendo clasificación')
            return None


def save_confusion_matrix(confusion_matrix,
                          classes,
                          filename,
                          normalize=True):
    # Adaptado de: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    cmap = pyplot.get_cmap('Blues')
    pyplot.figure(figsize=(8, 6))
    pyplot.imshow(confusion_matrix, interpolation='nearest',
                  cmap=cmap, aspect='auto')
    pyplot.title('Matriz de confusión')
    pyplot.colorbar()

    if classes is not None:
        tick_marks = numpy.arange(len(classes))
        pyplot.xticks(tick_marks, classes, rotation=45)
        pyplot.yticks(tick_marks, classes)

    thresh = confusion_matrix.max() / 2
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        pyplot.text(j, i, "{:,}".format(confusion_matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")

    pyplot.ylabel('Clase correcta')
    pyplot.xlabel('Clase predecida')
    pyplot.tight_layout()
    pyplot.savefig(filename)
    pyplot.close(filename)


def report_format(report):
    report_data = "<table>"
    lines = report.split('\n')
    report_data += "<tr>"
    report_data += "<th>Clase</th>"
    report_data += "<th>precision</th>"
    report_data += "<th>recall</th>"
    report_data += "<th>f1</th>"
    report_data += "<th>Cantidad de datos</th>"
    report_data += "</tr>"
    for line in lines[2:]:
        row_data = line.split('      ')
        if len(row_data) == 5:
            row_data[0] = row_data[0].replace(
                "avg / total", "<strong>promedio/total</strong>", 1)
            row = "<tr>"
            row += ("<td>" + row_data[0] + "</td>")
            row += ("<td>" + row_data[1] + "</td>")
            row += ("<td>" + row_data[2] + "</td>")
            row += ("<td>" + row_data[3] + "</td>")
            row += ("<td>" + row_data[4] + "</td>")
            row += "</tr>"
            report_data += row
        else:
            print('Registro invalido: ' + str(row_data))
    report_data += "</table>"
    return report_data
