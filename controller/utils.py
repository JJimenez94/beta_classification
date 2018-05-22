import os.path
import time
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
            elif model == "KNN":
                knn_chain = initialized_classifiers.trainKNN(
                    x_train, y_train, x_test, y_test)
                result.append("<p>")
                result.append(createDinamycHTML(knn_chain))
                result.append("</p>")
            elif model == "neural_net":
                ann_chain = initialized_classifiers.trainANN(
                    x_train, y_train, x_test, y_test)
                result.append("<p>")
                result.append(createDinamycHTML(ann_chain))
                result.append("</p>")
    if len(result) >= 1:
        result.append(
            "<center>" "<a href='/clasificar' class='button alt'>Clasificar datos</a>" "</center>")
    return result


def classifyModels(models_dict, ext, data_col):
    initialized_classifiers = classifiers()
    persistence_manager = uploader(ext)
    X = persistence_manager.uploadProductionDataset(data_col)
    result = {}
    for model in models_dict:
        if models_dict[model] == True:
            start_time = time.time()
            print("Se va a entrenar: " + model)
            if model == "naive":
                y = initialized_classifiers.classifyNaive(X)
                end_time = time.time()
                if y is not None:
                    persistence_manager.createResponse(y, "naive_bayes", data_col)
                    result['naive_bayes'] = end_time - start_time
            elif model == "decision_tree":
                y = initialized_classifiers.classifyDT(X)
                end_time = time.time()
                if y is not None:
                    persistence_manager.createResponse(y, "decision_tree", data_col)
                    result['decision_tree'] = end_time - start_time
            elif model == "svm":
                y = initialized_classifiers.classifySVM(X)
                end_time = time.time()
                if y is not None:
                    persistence_manager.createResponse(y, "SVM", data_col)
                    result['SVM'] = end_time - start_time
            elif model == "KNN":
                y = initialized_classifiers.classifyKNN(X)
                end_time = time.time()
                if y is not None:
                    persistence_manager.createResponse(y, "KNN", data_col)
                    result['KNN'] = end_time - start_time
            elif model == "neural_net":
                y = initialized_classifiers.classifyANN(X)
                end_time = time.time()
                if y is not None:
                    persistence_manager.createResponse(y, "ANN", data_col)
                    result['ANN'] = end_time - start_time
    return result


def clearFiles(filename, path):
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath):
        os.remove(filepath)


def createDinamycHTML(body):
    return "".join(body)
