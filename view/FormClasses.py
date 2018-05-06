from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import BooleanField

class AlgorithmForm(FlaskForm):
    naive = BooleanField('Clasificador ingenuo de Bayes (Naive Bayes)')
    svm = BooleanField('Máquinas de soporte vectorial (SVM)')
    ann = BooleanField('Redes Neuronales Artificiales (ANN)')
    km = BooleanField('K-Means')
    dt = BooleanField('Árboles de decisión')
    dataset = FileField('Seleccione el dataset: ', validators=[FileRequired()])