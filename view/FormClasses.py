from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import BooleanField

class DatasetForm(FlaskForm):
    dataset = FileField('Dataset', validators=[FileRequired()])

class AlgorithmForm(FlaskForm):
    naive = BooleanField('Naive', default="")
    svm = BooleanField('SVM', default="")
    ann = BooleanField('ANN', default="")
    km = BooleanField('KM', default="")
    dt = BooleanField('DT', default="")