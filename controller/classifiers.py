import itertools
import matplotlib.pyplot as pyplot
import numpy
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

    def testNaive(self, x, expected):
        labels = expected.drop_duplicates()        
        predicted = self.nb_classifier.predict(x)
        # Se muestran las metricas resultantes del entrenamiento
        result = metrics.classification_report(expected, predicted)        
        result = report_format(result)
        save_confusion_matrix(metrics.confusion_matrix(expected, predicted)
                            , labels
                            , "static/naive.png"
                            , False)
        filename = self.model_folders + self.naive_model
        elements = ["<h3> Reporte de resultados para el  clasificador ingenuo de Bayes </h3> <br>"
                , result
                , "<br>"
                , "<center>"
                , "<img src='static/naive.png' alt='Matriz de confusión clasificador ingenuo de Bayes'>"
                , "</center> <br>"
                , "Para descargar el modelo clasificado haga click: "
                , "<a href=" + filename + ">En este enlace</a>"]
        del labels
        return elements

    def trainNaive(self, x_train, y_train, x_test, y_test, user_alpha=1.0, user_binarize=0.0):
        if user_alpha != 1.0 or  user_binarize != 0.0:
            print('Redefiniendo al clasificador Bayesiano con parametros ingresados')
            self.nb_classifier = BernoulliNB(alpha=user_alpha, binarize=user_binarize)
        self.nb_classifier.fit(x_train, y_train)
        print(self.nb_classifier)
        filename = self.model_folders + self.naive_model
        pickle.dump(self.nb_classifier, open(filename, 'wb'))
        return self.testNaive(x_test, y_test)

    def classifyNaive(self, x):
        filename = self.model_folders + self.naive_model
        if os.path.exists(filename):
            loaded_naive = pickle.load(open(filename, 'rb'))
            return loaded_naive.predict(x)
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
    pyplot.imshow(confusion_matrix, interpolation='nearest', cmap=cmap, aspect='auto')
    pyplot.title('Matriz de confusión')
    pyplot.colorbar()

    if classes is not None:
        tick_marks = numpy.arange(len(classes))
        pyplot.xticks(tick_marks, classes, rotation=45)
        pyplot.yticks(tick_marks, classes)

    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, numpy.newaxis]

    thresh = confusion_matrix.max() / 1.5 if normalize else confusion_matrix.max() / 2
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        if normalize:
            pyplot.text(j, i, "{:0.4f}".format(confusion_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
        else:
            pyplot.text(j, i, "{:,}".format(confusion_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    pyplot.ylabel('Clase correcta')
    pyplot.xlabel('Clase predecida')    
    pyplot.ioff()
    pyplot.tight_layout()
    pyplot.savefig(filename)

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
            row_data[0] = row_data[0].replace("avg / total"
                , "<strong>promedio/total</strong>", 1)
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