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
    x_train, y_train, x_test, y_test = persistence_manager.uploadDataset(
        data_col, class_col)
    result = []
    for model in models_dict:
        if models_dict[model] == True:
            print("Se va a entrenar: " + model)
            if model == "naive":
                naives_chain = initialized_classifiers.trainNaive(
                    x_train, y_train, x_test, y_test)
                result.append("<p>")
                result.append(createDinamycHTML(naives_chain))
                result.append("</p>")
            elif model == "decision_tree":
                dt_chain = initialized_classifiers.trainDT(
                    x_train, y_train, x_test, y_test)
                result.append("<p>")
                result.append(createDinamycHTML(dt_chain))
                result.append("</p>")
            elif model == "svm":
                svm_chain = initialized_classifiers.trainSVM(
                    x_train, y_train, x_test, y_test)
                result.append("<p>")
                result.append(createDinamycHTML(svm_chain))
                result.append("</p>")
    return result


def clearFiles(filename, path):
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath):
        os.remove(filepath)


def createDinamycHTML(body):
    return "".join(body)
