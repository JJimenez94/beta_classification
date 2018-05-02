from view.FormClasses import DatasetForm, AlgorithmForm
from flask import Flask, render_template, request, flash
from flask_uploads import UploadSet, configure_uploads
import os.path

app = Flask(__name__)
app.debug=True
app.config.update(dict(
    SECRET_KEY="secretkey",
    UPLOADS_DEFAULT_DEST="uploads"   
))

datasets = UploadSet("datasets", ("txt","arff","csv","xls","xlsx"))
configure_uploads(app, datasets)

def changeName(currentFileName, operation):
    index = currentFileName.find('.')
    if (index != -1):
        newName = currentFileName[index:]
        if operation == "entrenar":
            return "train" + newName
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
    if request.method == "POST" and 'fileSelector' in request.files:
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
    return render_template("algorithm_layout.html", operation_type="entrenar")

@app.route("/clasificar", methods=["GET", "POST"])
def classificate():
    if request.method == "POST"  and 'fileSelector' in request.files:
        uploadFile =  request.files['fileSelector']
        temporal = changeName(uploadFile.filename, "clasificar")
        if uploadFile.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if temporal != None:
            clearFiles(temporal)
            uploadFile.filename = temporal
            datasets.save(uploadFile)
        return render_template("result.html", operation_type="clasificar")
    return render_template("algorithm_layout.html", operation_type="clasificar")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == "__main__":
    app.run()