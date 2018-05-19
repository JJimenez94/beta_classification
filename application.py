import os.path
from controller.utils import getFileExt, changeName, trainModels, clearFiles, createDinamycHTML
from flask import Flask, render_template, request, flash, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, send_from_directory
from view.FormClasses import AlgorithmForm, ModelForm, TrainForm

app = Flask(__name__)
app.debug=True
app.config.update(dict(
    SECRET_KEY="secretkey",
    UPLOADS_DEFAULT_DEST="uploads"   
))

datasets = UploadSet("datasets", ("txt","csv","xls","xlsx"))
models = UploadSet("models", ("sav"))
configure_uploads(app, datasets)
configure_uploads(app, models)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/conoce_beta")
def about():
    return render_template("about.html")

@app.route("/entrenar_algoritmos", methods=["GET", "POST"])
def train_dataset():
    form = TrainForm()
    if request.method == "POST" and form.validate_on_submit():
        choice = form.choicer.data        
        if choice == "train":
            return redirect(url_for('train'))
        else:
            return redirect(url_for('load_model'))
    return render_template("choicer.html", form=form)

@app.route("/modelo", methods=["GET", "POST"])
def load_model():
    form = ModelForm()
    if request.method == "POST" and form.validate_on_submit():        
        choice = form.choicer.data
        path = os.path.join(app.config['UPLOADS_DEFAULT_DEST']
                            , "models/")
        uploadFile = request.files['model']
        ext = getFileExt(uploadFile.filename)
        if choice == 'naive':
            complement = "ingenuo de Bayes"
            uploadFile.filename = "trained_naive" + ext
        elif choice == 'svm':
            complement = "maquinas de soporte vectorial"
            uploadFile.filename = "trained_svm" + ext
        elif choice == 'ann':
            complement = "red neuronal artificial"
            uploadFile.filename = "trained_ann" + ext
        elif choice == 'knn':
            complement = "K-Nearest Neighbors"
            uploadFile.filename = "trained_knn" + ext
        else:
            uploadFile.filename = "trained_dt" + ext
        clearFiles(uploadFile.filename, path)
        models.save(uploadFile)
        body_elements = ["<center> <p>"
                , "El modelo entrenado para el clasificador: <strong>"
                , complement
                , "</strong>"
                , " fue cargado correctamente.</p>"                
                , "<a href='/clasificar' class='button alt'>Clasificar datos</a>"
                , "</center>"
        ]        
        print('Archivo: ' + uploadFile.filename + ' cargado correctamente')
        return render_template("result.html", operation_type="cargar modelo entrenado"
                    , body=createDinamycHTML(body_elements))
    return render_template("model_loader.html", form=form)

@app.route("/dataset", methods=["GET", "POST"])
def train():
    form = AlgorithmForm()
    if request.method == "POST" and form.validate_on_submit():
        nb = form.naive.data
        svm = form.svm.data
        ann = form.ann.data
        knn = form.knn.data
        dt = form.dt.data
        classes = form.classes.data
        text = form.text.data
        if (nb or svm or ann or knn or dt):
            algorithms = {'naive':nb
                , 'svm':svm
                , 'neural_net':ann
                , 'KNN':knn
                , 'decision_tree':dt}
            uploadFile = request.files['dataset']
            newFileName = changeName(uploadFile.filename, "entrenar")
            if newFileName != None:
                ext = getFileExt(newFileName)
                path = os.path.join(app.config['UPLOADS_DEFAULT_DEST']
                            , "datasets/")
                clearFiles(newFileName, path)
                uploadFile.filename = newFileName
                datasets.save(uploadFile)
                print('Archivo: ' + newFileName + ' cargado correctamente')
            del uploadFile, newFileName,nb, svm, ann, knn, dt            
            return render_template("result.html", operation_type="entrenar"
                    , body=createDinamycHTML(trainModels(algorithms, ext, text, classes)))
        else:
            print("Debe seleccionar al menos un algoritmo, por favor intente de nuevo")        
    return render_template("algorithm_layout.html", operation_type="entrenar", form=form)

@app.route("/clasificar", methods=["GET", "POST"])
def classificate():
    # operation_type="clasificar"
    pass

@app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(app.root_path, app.config['UPLOADS_DEFAULT_DEST'])
    return send_from_directory(directory=uploads, filename=filename)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', error_type=404), 404

@app.errorhandler(500)
def error(e):
    return render_template('404.html', error_type=500), 500

if __name__ == "__main__":
    app.run()