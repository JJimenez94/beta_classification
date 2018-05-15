from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField, FileRequired
from wtforms import BooleanField, StringField
from wtforms.validators import DataRequired

class AlgorithmForm(FlaskForm):
    naive = BooleanField('Clasificador ingenuo de Bayes (Naive Bayes)')
    svm = BooleanField('Máquinas de soporte vectorial (SVM)')
    ann = BooleanField('Redes Neuronales Artificiales (ANN)')
    km = BooleanField('K-Means')
    dt = BooleanField('Árboles de decisión')
    classes = StringField('Nombre de la columna que contiene las clases: ', validators=[DataRequired(message="Valor requerido")])
    text = StringField('Nombre de la columna que contiene el texto: ', validators=[DataRequired(message="Valor requerido")])
    dataset = FileField('Seleccione el dataset: ', validators=[FileRequired(message="Archivo requerido"), 
                                                    FileAllowed(["txt","csv","xls","xlsx"]
                                                    , "solamente se permiten datasets")])