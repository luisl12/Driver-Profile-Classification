"""
modeling.decision_tree
----------------------------

This module provides supervised learning with decision trees
"""

# append the path of the parent directory
import sys
sys.path.append("..")

# packages
from collections import Counter
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from skl2onnx import convert_sklearn, update_registered_converter, to_onnx
from skl2onnx.common.data_types import FloatTensorType, guess_numpy_type
from onnxmltools.convert.xgboost.operator_converters.XGBoost import (
    convert_xgboost)
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from skl2onnx.algebra.onnx_operator import OnnxSubEstimator
from skl2onnx.algebra.onnx_ops import OnnxMatMul, OnnxSub
import onnxruntime as rt
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold, train_test_split
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
    det_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn2pmml import sklearn2pmml, make_pmml_pipeline
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# local
from pre_process import (
    read_csv_file,
    normalize_by_distance
)



def find_best_estimator(data, target, algorithm, grid, cv=10, path=None):
    scores = ['accuracy', 'precision', 'recall', 'f1']
    refit_score = 'f1'

    kfold = StratifiedKFold(n_splits=cv, shuffle=True)
    clf = GridSearchCV(algorithm, param_grid=grid, cv=kfold, scoring=scores, refit=refit_score, n_jobs=-1, return_train_score=True)
    clf.fit(data, target)

    best_score = clf.best_score_
    best_params = clf.best_params_
    best_estimator = clf.best_estimator_

    print('Train set - Accuracy:', clf.cv_results_['mean_train_accuracy'][:cv])
    print('Test set - Accuracy:', clf.cv_results_['mean_test_accuracy'][:cv], '\n')
    print('Train set - Precision:', clf.cv_results_['mean_train_precision'][:cv])
    print('Test set - Precision:', clf.cv_results_['mean_test_precision'][:cv], '\n')
    print('Train set - Recall:', clf.cv_results_['mean_train_recall'][:cv])
    print('Test set - Recall:', clf.cv_results_['mean_test_recall'][:cv], '\n')
    print('Train set - F1:', clf.cv_results_['mean_train_f1'][:cv])
    print('Test set - F1:', clf.cv_results_['mean_test_f1'][:cv], '\n')
    print('Best score {}:'.format(refit_score), best_score, 'with best params:', best_params)
    print('Best estimator:', best_estimator)

    if path:
        # save model
        with open(path + '.pickle', 'wb') as f:
            pickle.dump(best_estimator, f)


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


def nested_cross_validation(data, target, algorithm, inner_n_splits, outter_n_splits, grid):

    inner_cv = StratifiedKFold(n_splits=inner_n_splits, shuffle=True)
    outer_cv = StratifiedKFold(n_splits=outter_n_splits, shuffle=True)

    clf = GridSearchCV(estimator=algorithm, param_grid=grid, cv=inner_cv, scoring='f1', n_jobs=-1)

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
        'accuracy': 'accuracy',  # make_scorer(get_confusion_matrix_with_acc),
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'AUC': 'roc_auc'
    }

    scores = cross_validate(clf, X=data, y=target, cv=outer_cv, scoring=scoring, n_jobs=-1)
    return scores, confusion_matrixs, classification_reports, curves


def cross_validation(data, target, algorithm, cv=10, path=None):
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'AUC': 'roc_auc'
    }
    scores = cross_validate(algorithm, X=data, y=target, cv=kfold, scoring=scoring, n_jobs=-1)
    with open(path + 'evaluation.txt', 'w') as f:
        for s in scores:
            f.write("{} mean: {} with a standard deviation: {} \n".format(s, scores[s].mean(), scores[s].std()))


def evaluate_cross_validation(scores, confusion_matrixs, classification_reports, curves, path=None, show=False):

    if path:
        with open(path + 'evaluation.txt', 'w') as f:
            for s in scores:
                f.write("{} mean: {} with a standard deviation: {} \n".format(s, scores[s].mean(), scores[s].std()))

        # with open(path + 'classification_reports.txt', 'w') as f:
        #     for cr in classification_reports:
        #         print(cr)
        #         f.write(str(cr))
        #         f.write('\n')
    
    if show:
        print('Scores:', scores)
        for s in scores:
            print("{} mean: {} with a standard deviation: {}".format(s, scores[s].mean(), scores[s].std()))


# def save_model(model, name, path):
#     if is_xgb:
#         update_registered_converter(
#             xgb.XGBClassifier, 'XGBoostXGBClassifier',
#             calculate_linear_classifier_output_shapes, convert_xgboost,
#             options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
            
#         initial_type = [('float_input', FloatTensorType([None, n_features]))]
#         opset = {'': 12, 'ai.onnx.ml': 2}
#         onx = convert_sklearn(model, initial_types=initial_type, target_opset=opset)
        
#     else:
#         initial_type = [('float_input', FloatTensorType([None, n_features]))]
#         # onx = convert_sklearn(model, initial_types=initial_type)
#         onx = to_onnx(model, X=training_set.to_numpy())

#     with open(name + '.onnx', 'wb') as f:
#         f.write(onx.SerializeToString())


def predict_profile(path, x_test):
    # sess = rt.InferenceSession(model_name + '.onnx')
    # input_name = sess.get_inputs()[0].name
    # label_name = sess.get_outputs()[0].name
    # # pred_onx = sess.run([label_name], {input_name: data.to_numpy()})[0]
    # pred_onx = sess.run([label_name], {input_name: data.to_numpy().astype(np.float32)})[0]
    y_pred = None
    with open(path + '.pickle', 'rb') as f:
        model = pickle.load(f)
        y_pred = model.predict(x_test)
    return y_pred


def evaluate_predictions(y_true, y_pred, path=None, evaluation_name='evaluation', show=False):

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(y_true, y_pred)
    acc_score = accuracy_score(y_true, y_pred)
    prec_score = precision_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)

    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(cmap=plt.cm.Blues)

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    rcd = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    rcd.plot()

    prec, rec, _ = precision_recall_curve(y_test, y_pred)
    prd = PrecisionRecallDisplay(precision=prec, recall=rec)
    prd.plot()

    if path:
        with open(path + evaluation_name + '.txt'.format(), 'w') as f:
            f.write("Accuracy: {} \n".format(acc_score))
            f.write("Precision: {} \n".format(prec_score))
            f.write("Recall: {} \n".format(rec_score))
            f.write("F1: {} \n".format(f1score))
            f.write("Number of TP: {} \n".format(tp))
            f.write("Number of TN: {} \n".format(tn))
            f.write("Number of FP: {} \n".format(fp))
            f.write("Number of FN: {} \n \n".format(fn))
            f.write(report)
            plt.savefig(path + 'confusion_matrix')

    if show:
        print(report)
        print('Accuracy:', accuracy_score(y_true, y_pred))
        print('Precision:', precision_score(y_true, y_pred))
        print('Recall:', recall_score(y_true, y_pred))
        print('F1:', f1_score(y_true, y_pred))
        print('Number of TP:', tp)
        print('Number of TN:', tn)
        print('Number of FP:', fp)
        print('Number of FN:', fn)
        plt.show()


def show_dataset_info(df):
    print('Dataset shape:', df.shape)
    print('Number of trips belonging to class 0:', len(df[df['target'] == 0]))
    print('Number of trips belonging to class 1:', len(df[df['target'] == 1]))



if __name__ == "__main__":

    # read first and second dataset and cocatenate both
    df = read_csv_file('../datasets/supervised/trips_kmeans')

    print('------------------ DATASET INFO ------------------ \n')
    show_dataset_info(df)

    data = df.drop('target', axis=1)
    target = df['target']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, stratify=target, random_state=42)
    print('Train balance after split:', Counter(y_train))
    print('Test balance after split:', Counter(y_test))

    # --------------------------------- DEFINE PIPELINE --------------------------------- #

    # define classifiers
    dtc = DecisionTreeClassifier()
    xgbc = xgb.XGBClassifier(objective="binary:logistic")
    rfc = RandomForestClassifier()

    # define param grid for each classifier
    dtc_grid = {
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [6, 8, 10, 12],  # very important for overfitting
        'clf__min_samples_split': [0.05, 0.1, 0.15, 0.2],  # very important for overfitting
        'clf__min_samples_leaf': [0.05, 0.1, 0.15, 0.2],  # very important for overfitting
        # 'clf__max_features': [0.2, 0.6, 1.0],
        'over__k_neighbors': [4, 5, 6, 8, 10]
    }         
    rfc_grid = {
        'clf__n_estimators':[100, 150, 200, 250],  # just pick a high number
        'clf__criterion': ['gini', 'entropy'],
        # 'clf__max_depth': [8, 10, 12],  # not very important in random forest
        'clf__min_samples_split': [0.02, 0.05, 0.1, 0.15, 0.2],
        # 'clf__min_samples_leaf': [1, 2, 4],  # not very important in random forest
        'over__k_neighbors': [4, 5, 6, 8]
    }
    xgbc_grid = {
        'clf__n_estimators':[100, 150, 200, 250],
        'clf__max_depth':[3, 4, 6, 8, 10],
        'clf__learning_rate': [0.01, 0.05, 0.1],
        # 'clf__gamma': [0],
        'clf__colsample_bytree': [0.3, 0.5, 0.7],
        # 'clf__subsample': [0.2, 0.5, 0.8, 1.0],
        # 'clf__min_child_weight': [0.2, 0.4, 0.6, 0.8]
        'over__k_neighbors': [4, 5, 6, 8]
    }

    algs = {
        # 'decision_tree': dtc,
        # 'random_forest': rfc,
        'xgboost': xgbc
    }
    grids = {
        # 'decision_tree': dtc_grid,
        # 'random_forest': rfc_grid,
        'xgboost': xgbc_grid
    }

    # define number of cross validations
    parameter_cv = 5
    cv = 10

    for a in algs:

        # X_train = normalize_by_distance(X_train)

        # define imblearn pipeline
        pipeline = IMBPipeline(steps=[
            ('norm', FunctionTransformer(normalize_by_distance)),
            ('over', SMOTE(sampling_strategy=0.3)),
            ('under', RandomUnderSampler(sampling_strategy=0.5)),
            ('clf', algs[a])
        ])

        # ----------------------------- NESTED CROSS VALIDATION ------------------------------ #
        print('# ---------------------- NESTED CROSS VALIDATION ---------------------- # \n')

        # scores, confusion_matrixs, classification_reports, curves = nested_cross_validation(X_train, y_train, pipeline, parameter_cv, cv, grids[a])
        # path = './images/supervised/{}/train/with_resample/'.format(a)
        # evaluate_cross_validation(scores, confusion_matrixs, classification_reports, curves, path=path, show=False)

        # --------------------------- TRAIN MODEL WITH BEST PARAMS --------------------------- #
        print('# -------------------- TRAIN MODEL WITH BEST PARAMS -------------------- # \n')

        # path = './images/supervised/{}/best_resample/borderline_'.format(a)
        # cross_validation(X_train, y_train, pipeline, cv=cv, path=path)
        estim_path = './models/{}_model'.format(a)
        find_best_estimator(X_train, y_train, pipeline, grids[a], cv=10, path=estim_path)
        # tree_structure(model)

        # ------------------------------------ TEST MODEL ------------------------------------ #
        print('# ----------------------------- TEST MODEL ----------------------------- # \n')

        y_pred = predict_profile(estim_path, X_test)
        print(y_pred)
        
        path = './images/supervised/{}/best_resample/over_under/'.format(a)
        evaluate_predictions(y_test, y_pred, path=None, show=True)
