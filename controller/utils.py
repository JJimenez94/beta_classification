import os.path
from controller.classifiers import classifiers
from persistence.persistenceManager import uploader

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
            

def clearFiles(filename, path):
    filepath = os.path.join(path
                , filename)
    if os.path.exists(filepath):
        os.remove(filepath)

def createDinamycHTML(body):    
    return "".join(body)