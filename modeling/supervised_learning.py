"""
modeling.decision_tree
----------------------------

This module provides supervised learning with decision trees
"""

# append the path of the parent directory
import sys
sys.path.append("..")

# packages
from itertools import cycle
import joblib
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (
    make_scorer, 
    confusion_matrix, 
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.preprocessing import label_binarize, FunctionTransformer
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
# local
from pre_process import read_csv_file



def find_best_estimator(data, target, algorithm, grid, cv=10, is_multi_class=False, path=None):
    scores = ['accuracy', 'precision', 'recall', 'f1']
    refit_score = 'f1'
    if is_multi_class:
        scores = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        refit_score = 'f1_weighted'

    kfold = StratifiedKFold(n_splits=cv, shuffle=True)
    clf = GridSearchCV(algorithm, param_grid=grid, cv=kfold, scoring=scores, refit=refit_score, n_jobs=-1, return_train_score=True)
    clf.fit(data, target)

    best_score = clf.best_score_
    best_params = clf.best_params_
    best_estimator = clf.best_estimator_

    print('Train set - Accuracy:', clf.cv_results_['mean_train_accuracy'][:cv])
    print('Test set - Accuracy:', clf.cv_results_['mean_test_accuracy'][:cv], '\n')
    print('Train set - Precision:', clf.cv_results_['mean_train_' + scores[1]][:cv])
    print('Test set - Precision:', clf.cv_results_['mean_test_' + scores[1]][:cv], '\n')
    print('Train set - Recall:', clf.cv_results_['mean_train_' + scores[2]][:cv])
    print('Test set - Recall:', clf.cv_results_['mean_test_' + scores[2]][:cv], '\n')
    print('Train set - F1:', clf.cv_results_['mean_train_' + scores[3]][:cv])
    print('Test set - F1:', clf.cv_results_['mean_test_' + scores[3]][:cv], '\n')
    print('Best score {}:'.format(refit_score), best_score, 'with best params:', best_params)
    print('Best estimator:', best_estimator)

    if path:
        # save model
        joblib.dump(best_estimator, path + '.joblib')

    return best_estimator


def tree_structure(model):
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node.".format(
                    space=node_depth[i] * "\t", node=i
                )
            )
        else:
            print(
                "{space}node={node} is a split node: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                )
            )


def nested_cross_validation(data, target, algorithm, inner_n_splits, outter_n_splits, grid, is_multi_class=False):

    inner_cv = StratifiedKFold(n_splits=inner_n_splits, shuffle=True)
    outer_cv = StratifiedKFold(n_splits=outter_n_splits, shuffle=True)

    scoring = 'f1' if not is_multi_class else 'f1_weighted'
    clf = GridSearchCV(estimator=algorithm, param_grid=grid, cv=inner_cv, scoring=scoring, n_jobs=-1)

    confusion_matrixs = []
    classification_reports = []
    curves = {
        'roc': [],
        'det': [],
        'prec_rec': []
    }

    # def get_confusion_matrix_with_acc(y_true, y_pred):
    #     confusion_matrixs.append(confusion_matrix(y_true, y_pred, labels=np.unique(target)))  # , normalize='true'
    #     classification_reports.append(classification_report(y_true, y_pred))
    #     curves['roc'].append(roc_curve(y_true, y_pred))
    #     curves['det'].append(det_curve(y_true, y_pred))
    #     curves['prec_rec'].append(precision_recall_curve(y_true, y_pred))
    #     return accuracy_score(y_true, y_pred)

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision' if not is_multi_class else 'precision_weighted',
        'recall': 'recall' if not is_multi_class else 'recall_weighted',
        'f1': 'f1' if not is_multi_class else 'f1_weighted',
        'AUC': 'roc_auc' if not is_multi_class else 'roc_auc_ovo_weighted' 
    }

    scores = cross_validate(clf, X=data, y=target, cv=outer_cv, scoring=scoring, return_train_score=True, n_jobs=-1)
    return scores


def cross_validation(data, target, algorithm, cv=10, is_multi_class=False, path=None):
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision' if not is_multi_class else 'precision_weighted',
        'recall': 'recall' if not is_multi_class else 'recall_weighted',
        'f1': 'f1' if not is_multi_class else 'f1_weighted',
        'AUC': 'roc_auc' if not is_multi_class else 'roc_auc_ovr'
    }
    scores = cross_validate(algorithm, X=data, y=target, cv=kfold, scoring=scoring, n_jobs=-1)
    with open(path + 'evaluation.txt', 'w') as f:
        for s in scores:
            f.write("{} mean: {} with a standard deviation: {} \n".format(s, scores[s].mean(), scores[s].std()))


def evaluate_cross_validation(scores, path=None, show=False):

    if path:
        with open(path + 'evaluation.txt', 'w') as f:
            for s in scores:
                f.write("{} mean: {} with a standard deviation: {} \n".format(s, scores[s].mean(), scores[s].std()))
    
    if show:
        print('Scores:', scores)
        for s in scores:
            print("{} mean: {} with a standard deviation: {}".format(s, scores[s].mean(), scores[s].std()))


def predict_profile(path, x_test, is_multi_class=False, is_svm=False):
    y_pred = y_pred_proba = None

    with open(path + '.joblib', 'rb') as f:
        model = joblib.load(f)
        y_pred = model.predict(x_test)
        if is_svm:
            y_pred_proba = model.decision_function(x_test)
        else:
            if is_multi_class:
                y_pred_proba = model.predict_proba(x_test)
            else:
                y_pred_proba = model.predict_proba(x_test)[:, 1]

    return y_pred, y_pred_proba


def evaluate_predictions(y_true, y_pred, y_pred_proba, is_multi_class=False, path=None, show=False):

    cm = confusion_matrix(y_true, y_pred)
    if not is_multi_class:
        tn, fp, fn, tp = cm.ravel()
    report = classification_report(y_true, y_pred)
    acc_score = accuracy_score(y_true, y_pred)
    prec_score = precision_score(y_true, y_pred, average='weighted' if is_multi_class else 'binary')
    rec_score = recall_score(y_true, y_pred, average='weighted' if is_multi_class else 'binary')
    f1score = f1_score(y_true, y_pred, average='weighted' if is_multi_class else 'binary')

    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(cmap=plt.cm.Blues)
    if path:
        plt.savefig(path + 'confusion_matrix')

    if not is_multi_class:
        RocCurveDisplay.from_predictions(y_true, y_pred_proba)
        if path:
            plt.savefig(path + 'roc_curve')
        PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba)
        if path:
            plt.savefig(path + 'precision_recall_curve')
    else:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        prec = dict()
        rec = dict()
        # binarize ovr
        y_test = label_binarize(y_true, classes=np.unique(y_true))
        for i in range(len(np.unique(y_true))):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            prec[i], rec[i], _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])

        plt.figure()
        colors = cycle(["r", "g", "b"])
        for i, color in zip(range(len(np.unique(y_true))), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
            )
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        if path:
            plt.savefig(path + 'roc_curve')

        plt.figure()
        for i, color in zip(range(len(np.unique(y_true))), colors):
            plt.plot(
                rec[i],
                prec[i],
                color=color,
                lw=1.5,
                label="Precision-Recall curve of class {}".format(i),
            )
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower right")
        if path:
            plt.savefig(path + 'precision_recall_curve')  
        
    if path:
        with open(path + 'evaluation.txt'.format(), 'w') as f:
            f.write("Accuracy: {} \n".format(acc_score))
            f.write("Precision: {} \n".format(prec_score))
            f.write("Recall: {} \n".format(rec_score))
            f.write("F1: {} \n".format(f1score))
            if not is_multi_class:
                f.write("Number of TP: {} \n".format(tp))
                f.write("Number of TN: {} \n".format(tn))
                f.write("Number of FP: {} \n".format(fp))
                f.write("Number of FN: {} \n \n".format(fn))
            f.write(report)

    if show:
        print(report)
        print('Accuracy:', acc_score)
        print('Precision:', prec_score)
        print('Recall:', rec_score)
        print('F1:', f1score)
        if not is_multi_class:
            print('Number of TP:', tp)
            print('Number of TN:', tn)
            print('Number of FP:', fp)
            print('Number of FN:', fn)
        plt.show()


def show_dataset_info(df):
    classes = np.unique(df['target'])
    print('Dataset shape:', df.shape)
    for c in classes:
        print('Number of trips belonging to class {}:'.format(c), len(df[df['target'] == c]))


def normalize_by_distance(df):
    """
    Normalize dataset by trip distance.
    Each instance gets divided by trip distance.

    Args:
        df (pandas.DataFrame): Dataset

    Returns:
        pandas.DataFrame: Dataset normalized
    """
    trips = df.div(df['distance'], axis=0)
    trips = trips.drop(labels='distance', axis=1)
    trips = trips.drop(labels='duration', axis=1)
    return trips


if __name__ == "__main__":

    # read first and second dataset and cocatenate both
    df = read_csv_file('../datasets/supervised/trips_kmeans_agressive')

    print('------------------ DATASET INFO ------------------ \n')
    show_dataset_info(df)

    data = df.drop('target', axis=1)
    target = df['target']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, stratify=target, shuffle=True)
    print('Train balance after split:', Counter(y_train))
    print('Test balance after split:', Counter(y_test))

    # --------------------------------- DEFINE PIPELINE --------------------------------- #

    # define classifiers
    dtc = DecisionTreeClassifier()
    xgbc = xgb.XGBClassifier(objective="binary:logistic")
    rfc = RandomForestClassifier()
    svm = SVC(kernel="linear", probability=True)

    # define param grid for each classifier
    dtc_grid = {
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [6, 8, 10, 12],  # very important for overfitting
        'clf__min_samples_split': [0.05, 0.1, 0.15, 0.2],  # very important for overfitting
        'clf__min_samples_leaf': [0.05, 0.1, 0.15, 0.2],  # very important for overfitting
        # 'clf__max_features': [0.2, 0.6, 1.0],
        'over__n_neighbors': [4, 5, 6, 8]
    }         
    rfc_grid = {
        'clf__n_estimators':[100, 150, 200, 250],  # just pick a high number
        'clf__criterion': ['gini', 'entropy'],
        # 'clf__max_depth': [8, 10, 12],  # not very important in random forest
        'clf__min_samples_split': [0.02, 0.05, 0.1, 0.15, 0.2],
        # 'clf__min_samples_leaf': [1, 2, 4],  # not very important in random forest
        'over__n_neighbors': [4, 5, 6, 8]
    }
    xgbc_grid = {
        'clf__n_estimators':[150, 200, 250],
        'clf__max_depth':[6, 8], # [6, 8, 10]
        'clf__learning_rate': [0.01, 0.05, 0.1], 
        'clf__colsample_bytree': [0.3, 0.5, 0.7],
        # 'over__n_neighbors': [5, 6, 8]
    }
    svm_grid = {
        'clf__C':[0.001, 0.01, 0.1, 1, 10, 100],
        'over__n_neighbors': [4, 5]
    }

    algs = {
        # 'decision_tree': dtc,
        # 'random_forest': rfc,
        # 'xgboost': xgbc,
        'svm': svm
    }
    grids = {
        # 'decision_tree': dtc_grid,
        # 'random_forest': rfc_grid,
        # 'xgboost': xgbc_grid,
        'svm': svm_grid
    }

    # define number of cross validations
    parameter_cv = 10
    cv = 10

    for a in algs:

        # define imblearn pipeline
        pipeline = IMBPipeline(steps=[
            ('norm', FunctionTransformer(normalize_by_distance)),
            ('over', ADASYN()),
            # ('under', RandomUnderSampler(sampling_strategy=0.5)),
            ('red', PCA(0.99)),
            ('clf', algs[a])
        ])

        # ----------------------------- NESTED CROSS VALIDATION ------------------------------ #
        print('# ---------------------- NESTED CROSS VALIDATION {} ---------------------- # \n'.format(a))

        scores = nested_cross_validation(X_train, y_train, pipeline, parameter_cv, cv, grids[a], is_multi_class=True)
        path = './images/supervised/{}/train/multiclass/'.format(a)
        evaluate_cross_validation(scores, path=path, show=True)

        # --------------------------- TRAIN MODEL WITH BEST PARAMS --------------------------- #
        print('# -------------------- TRAIN MODEL WITH BEST PARAMS {} -------------------- # \n'.format(a))

        # path = './images/supervised/{}/best_resample/multiclass/smote_'.format(a)
        # cross_validation(X_train, y_train, pipeline, cv=cv, is_multi_class=True, path=path)
        # estim_path = './models/{}_model_pca_multi'.format(a)
        # find_best_estimator(X_train, y_train, pipeline, grids[a], cv=parameter_cv, is_multi_class=True, path=estim_path)

        # ------------------------------------ TEST MODEL ------------------------------------ #
        print('# ----------------------------- TEST MODEL {} ----------------------------- # \n'.format(a))

        # y_pred, y_pred_proba = predict_profile(estim_path, X_test, is_multi_class=True, is_svm=a=='svm')
        # path = './images/supervised/{}/test/multiclass/'.format(a)
        # evaluate_predictions(y_test, y_pred, y_pred_proba, is_multi_class=True, path=path, show=False)
