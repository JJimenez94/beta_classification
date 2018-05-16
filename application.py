import os.path
from controller.classifiers import classifiers
from flask import Flask, render_template, request, flash, redirect
from flask_uploads import UploadSet, configure_uploads
from persistence.persistenceManager import uploader
from view.FormClasses import AlgorithmForm

app = Flask(__name__)
app.debug=True
app.config.update(dict(
    SECRET_KEY="secretkey",
    UPLOADS_DEFAULT_DEST="uploads"   
))

datasets = UploadSet("datasets", ("txt","csv","xls","xlsx"))
configure_uploads(app, datasets)

def getFileExt(filename):
    index = filename.rfind('.')
    if (index != -1):
        return filename[index:]
    return None

def changeName(currentFileName, operation):
    ext = getFileExt(currentFileName)
    if ext != None:
        if operation == "entrenar":
            return "dataset" + ext
        elif operation == "clasificar":
            return "production" + ext
    return None

def trainModels(models_dict, ext, data_col, class_col):
    initialized_classifiers = classifiers()
    persistence_manager = uploader(ext)
    x_train, y_train, x_test, y_test = persistence_manager.uploadDataset(data_col, class_col)
    for model in models_dict:
        if models_dict[model] == True:
            print("Se va a entrenar: " + model)
            if model == "naive":
                initialized_classifiers.trainNaive(x_train, y_train, x_test, y_test)
            

def clearFiles(filename):
    filepath = os.path.join(
    os.path.join(
        app.config['UPLOADS_DEFAULT_DEST'],
        "datasets/"),
    filename)
    if os.path.exists(filepath):
        os.remove(filepath)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/conoce_beta")
def about():
    return render_template("about.html")

@app.route("/entrenar_algoritmos", methods=["GET", "POST"])
def train():
    form = AlgorithmForm()
    if request.method == "POST" and form.validate_on_submit():
        nb = form.naive.data
        svm = form.svm.data
        ann = form.ann.data
        km = form.km.data
        dt = form.dt.data
        classes = form.classes.data
        text = form.text.data
        if (nb or svm or ann or km or dt):
            algorithms = {'naive':nb
                , 'svm':svm
                , 'neural_net':ann
                , 'k_means':km
                , 'decision_tree':dt}
            uploadFile = request.files['dataset']
            newFileName = changeName(uploadFile.filename, "entrenar")
            ext = getFileExt(newFileName)
            if newFileName != None:
                clearFiles(newFileName)
                uploadFile.filename = newFileName
                datasets.save(uploadFile)
                print('Archivo: ' + newFileName + ' cargado correctamente')
        else:
            print("Debe seleccionar al menos un algoritmo, por favor intente de nuevo")
        del uploadFile, newFileName,nb, svm, ann, km, dt
        trainModels(algorithms, ext, text, classes)
        return render_template("result.html", operation_type="entrenar", dataset_type=ext
        , algorithms=algorithms)
    return render_template("algorithm_layout.html", operation_type="entrenar", form=form)

@app.route("/clasificar", methods=["GET", "POST"])
def classificate():
    form = AlgorithmForm()
    if request.method == "POST" and form.validate_on_submit():
        nb = form.naive.data
        svm = form.svm.data
        ann = form.ann.data
        km = form.km.data
        dt = form.dt.data
        if (nb or svm or ann or km or dt):
            algorithms = {'naive':nb
                , 'svm':svm
                , 'neural_net':ann
                , 'k_means':km
                , 'decision_tree':dt}            
            uploadFile = request.files['dataset']
            newFileName = changeName(uploadFile.filename, "clasificar")
            ext = getFileExt(newFileName)
            if newFileName != None:
                clearFiles(newFileName)
                uploadFile.filename = newFileName
                datasets.save(uploadFile)
                print('Archivo: ' + newFileName + ' cargado correctamente')
            else:
                print("Debe seleccionar al menos un algoritmo, por favor intente de nuevo")
            del uploadFile, newFileName,nb, svm, ann, km, dt
            return render_template("result.html", operation_type="clasificar"
                , dataset_type=ext, algorithms=algorithms)            
    return render_template("algorithm_layout.html", operation_type="clasificar", form=form)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', error_type=404), 404

@app.errorhandler(500)
def error(e):
    return render_template('404.html', error_type=500), 500

if __name__ == "__main__":
    app.run()