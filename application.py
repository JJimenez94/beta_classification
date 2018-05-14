from view.FormClasses import AlgorithmForm
from flask import Flask, render_template, request, flash, redirect
from flask_uploads import UploadSet, configure_uploads
import os.path

app = Flask(__name__)
app.debug=True
app.config.update(dict(
    SECRET_KEY="secretkey",
    UPLOADS_DEFAULT_DEST="uploads"   
))

datasets = UploadSet("datasets", ("txt","csv","xls","xlsx"))
configure_uploads(app, datasets)

def changeName(currentFileName, operation):
    index = currentFileName.rfind('.')
    if (index != -1):
        newName = currentFileName[index:]
        if operation == "entrenar":
            return "dataset" + newName
        elif operation == "clasificar":
            return "production" + newName
    return None

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
    form = AlgorithmForm(request.form)
    if request.method == "POST" and 'fileSelector' in request.files and form.validate():
        uploadFile =  request.files['fileSelector']
        temporal = changeName(uploadFile.filename, "entrenar")
        if uploadFile.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if temporal != None:
            clearFiles(temporal)
            uploadFile.filename = temporal
            datasets.save(uploadFile)
            flash('File ' + temporal + ' uploaded')
        return render_template("result.html", operation_type="entrenar")
    return render_template("algorithm_layout.html", operation_type="entrenar", form=form)

@app.route("/clasificar", methods=["GET", "POST"])
def classificate():
    form = AlgorithmForm(request.form)
    if request.method == "POST"  and 'fileSelector' in request.files:
        nb = form.naive.data
        svm = form.svm.data
        ann = form.ann.data
        km = form.km.data
        dt = form.dt.data
        print(str(nb))
        print(str(svm))
        print(str(ann))
        print(str(km))
        print(str(dt))
        if (nb or svm or ann or km or dt):            
            uploadFile =  request.files['fileSelector']
            temporal = changeName(uploadFile.filename, "clasificar")            
            if temporal != None:
                clearFiles(temporal)
                uploadFile.filename = temporal
                datasets.save(uploadFile)
            return render_template("result.html", operation_type="clasificar")
        else:
            flash('No selected algoritm')
            return redirect(request.url)
    return render_template("algorithm_layout.html", operation_type="clasificar", form=form)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == "__main__":
    app.run()