from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import BooleanField, RadioField, StringField
from wtforms.validators import DataRequired

class AlgorithmForm(FlaskForm):
    naive = BooleanField('Clasificador ingenuo de Bayes (Naive Bayes)')
    svm = BooleanField('Máquinas de soporte vectorial (SVM)')
    ann = BooleanField('Redes Neuronales Artificiales (ANN)')
    knn = BooleanField('K-Nearest Neighbors (KNN)')
    dt = BooleanField('Árboles de decisión')
    classes = StringField('Nombre de la columna que contiene las clases: ', validators=[DataRequired(message="Valor requerido")])
    text = StringField('Nombre de la columna que contiene el texto: ', validators=[DataRequired(message="Valor requerido")])
    dataset = FileField('Seleccione el dataset: '
                        , validators=[FileRequired(message="Archivo requerido")
                        , FileAllowed(["txt","csv","xls","xlsx"]
                        , "solamente se permiten datasets")])

class ModelForm(FlaskForm):
    choicer = RadioField(
        'Algoritmos disponibles: '
        , validators=[DataRequired(message="Valor requerido")]
        , choices=[('naive', 'Clasificador ingenuo de Bayes (Naive Bayes)')
            , ('svm', 'Máquinas de soporte vectorial (SVM)')
            , ('ann', 'Redes Neuronales Artificiales (ANN)')
            , ('knn', 'K-Nearest Neighbors (KNN)')
            , ('dt', 'Árboles de decisión')])
    model = FileField('Seleccione el archivo a cargar: '
                , validators=[FileRequired(message="Archivo requerido")
                , FileAllowed(["sav"]
                , "solamente se permiten archivos entregados por la aplicación")])

class TrainForm(FlaskForm):
    choicer = RadioField(
        '¿Que operación desea realizar?'
        , validators=[DataRequired(message="Valor requerido")]
        , choices=[('load', 'Cargar modelo previamente entrenado')
                    , ('train', 'Entrenar modelo con dataset')])