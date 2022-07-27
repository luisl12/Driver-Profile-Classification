"""
modeling.decision_tree
----------------------------

This module provides supervised learning with decision trees
"""

# append the path of the parent directory
import sys
sys.path.append("..")

# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
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
    det_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as IMBPipeline
from sklearn2pmml import sklearn2pmml, make_pmml_pipeline
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
# local
from pre_process import read_csv_file
from custom_missing_value_transformer import CustomMissingValueTransformer



def find_best_estimator(data, target, algorithm, grid, cv=10, path=None):
    scores = ['accuracy', 'recall']
    refit_score = 'recall'

    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)
    clf = GridSearchCV(algorithm, param_grid=grid, cv=kfold, scoring=scores, refit=refit_score, n_jobs=-1)
    clf.fit(data, target)

    best_score = clf.best_score_
    best_params = clf.best_params_
    best_estimator = clf.best_estimator_
    print('Accuracy:', clf.cv_results_['mean_test_accuracy'][clf.best_index_])
    print('Best score {}:'.format(refit_score), best_score, 'with best params:', best_params)
    print('Best estimator:', best_estimator)
    if path:
        # save tree
        plot_tree(best_estimator, filled=True)
        plt.title("Decision tree with best parameters")
        plt.savefig(path + '/tree.png')

    # print(export_text(best_estimator))

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


def nested_cross_validation(data, target, algorithm, inner_n_splits, outter_n_splits, grid):

    inner_cv = StratifiedKFold(n_splits=inner_n_splits, shuffle=True)
    outer_cv = StratifiedKFold(n_splits=outter_n_splits, shuffle=True)

    clf = GridSearchCV(estimator=algorithm, param_grid=grid, cv=inner_cv, scoring='f1', n_jobs=-1)
    # scores = cross_val_score(clf, X=data, y=target, cv=outer_cv)
    # print("{} accuracy with a standard deviation of {}".format(scores.mean(), scores.std()))

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

    # # confusion matrixs
    # for i, cm in enumerate(confusion_matrixs):
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(classification_reports[0].keys())[:2])
    #     disp.plot(cmap=plt.cm.Blues)
    #     if path:
    #         plt.savefig(path + 'conf_matrix_plots/confusion_matrix_{}.png'.format(i))
    #     if show:
    #         plt.show()

    # # roc curves
    # for i, roc in enumerate(curves['roc']):
    #     fpr, tpr, _ = roc
    #     disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=scores['test_AUC'][i], estimator_name='ROC fold {}'.format(i))
    #     disp.plot()
    #     if path:
    #         plt.savefig(path + 'roc_plots/roc_{}.png'.format(i))
    #     if show:
    #         plt.show()

    # # precision recall curves
    # for i, prec_rec in enumerate(curves['prec_rec']):
    #     prec, rec, _ = prec_rec
    #     disp = PrecisionRecallDisplay(precision=prec, recall=rec, estimator_name='PREC/REC fold {}'.format(i))
    #     disp.plot()
    #     if path:
    #         plt.savefig(path + 'prec_rec_plots/precision_recall_{}.png'.format(i))
    #     if show:
    #         plt.show()


def save_model(model, name, n_features, training_set, is_xgb):

    # if is_xgb:
    #     update_registered_converter(
    #         xgb.XGBClassifier, 'XGBoostXGBClassifier',
    #         calculate_linear_classifier_output_shapes, convert_xgboost,
    #         options={'nocl': [True, False], 'zipmap': [True, False, 'columns']})
            
    #     initial_type = [('float_input', FloatTensorType([None, n_features]))]
    #     opset = {'': 12, 'ai.onnx.ml': 2}
    #     onx = convert_sklearn(model, initial_types=initial_type, target_opset=opset)
        
    # else:
    #     initial_type = [('float_input', FloatTensorType([None, n_features]))]
    #     # onx = convert_sklearn(model, initial_types=initial_type)
    #     onx = to_onnx(model, X=training_set.to_numpy())

    # with open(name + '.onnx', 'wb') as f:
    #     f.write(onx.SerializeToString())
    print('')


def predict_profile(model_name, data):
    sess = rt.InferenceSession(model_name + '.onnx')
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    # pred_onx = sess.run([label_name], {input_name: data.to_numpy()})[0]
    pred_onx = sess.run([label_name], {input_name: data.to_numpy().astype(np.float32)})[0]
    return pred_onx


def evaluate_predictions(y_true, y_pred, path=None, evaluation_name='evaluation', show=False):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(y_true, y_pred)
    acc_score = accuracy_score(y_true, y_pred)
    prec_score = precision_score(y_true, y_pred)
    rec_score = recall_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

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
    df1 = read_csv_file('../datasets/supervised/trips_kmeans')
    df2 = read_csv_file('../datasets/supervised/trips_test')
    df = pd.concat([df1, df2])

    # remove start, and end
    df = df[(df.columns.difference([
        'trip_start', 'trip_end', 'light_mode'
    ], sort=False))]

    print('------------------ DATASET INFO ------------------ \n')
    show_dataset_info(df)

    data = df.drop('target', axis=1)
    target = df['target']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, stratify=target, random_state=42)

    # --------------------------------- DEFINE PIPELINE --------------------------------- #

    # define classifiers
    dtc = DecisionTreeClassifier()
    xgbc = xgb.XGBClassifier(objective="binary:logistic")
    rfc = RandomForestClassifier()

    # define param grid for each classifier
    dtc_grid = {
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [2, 4, 6, 8, 10],
        'clf__min_samples_split': [0.2, 0.4, 0.6, 0.8, 1.0],
        'clf__min_samples_leaf': [0.1, 0.2, 0.3, 0.4, 0.5],
        'clf__max_features': [0.2, 0.4, 0.6, 0.8, 1.0, 'sqrt', 'log2'],
        'over__k_neighbors': [2, 4, 6, 8, 10]
    }
            
    rfc_grid = {
        'clf__n_estimators':[50, 100, 150],
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': [2, 4, 6, 8, 10],
        'clf__min_samples_split': [0.2, 0.4, 0.6, 0.8, 1.0],
        'clf__min_samples_leaf': [0.1, 0.2, 0.3, 0.4, 0.5],
        'clf__max_features': [0.2, 0.4, 0.6, 0.8, 1.0, 'sqrt', 'log2', None],
        'over_smote__k_neighbors': [2, 4, 6, 8, 10]
    }
    xgbc_grid = {
        'clf__n_estimators':[50, 100, 150],
        'clf__subsample': [0.2, 0.5, 0.8, 1.0],
        'clf__max_depth':[2, 4, 6, 8],
        'clf__learning_rate': [0.05, 0.1, 0.2],
        'clf__gamma': [0, 0.5, 1],
        'clf__min_child_weight': [0.2, 0.4, 0.6, 0.8]
        # 'svm_smote__k_neighbors': [2, 4, 6, 8, 10]
    }

    algs = {
        'decision_tree': dtc,
        # 'random_forest': rfc,
        # 'xgboost': xgbc
    }
    grids = {
        'decision_tree': dtc_grid,
        # 'random_forest': rfc_grid,
        # 'xgboost': xgbc_grid
    }

    # define number of cross validations
    parameter_cv = 5
    cv = 10

    for a in algs:

        # define imblearn pipeline
        pipeline = IMBPipeline(steps=[
            ('over', BorderlineSMOTE()),
            # ('under', RandomUnderSampler(sampling_strategy=0.5)),
            ('clf', algs[a])
        ])

        # ----------------------------- NESTED CROSS VALIDATION ------------------------------ # 

        # scores, confusion_matrixs, classification_reports, curves = nested_cross_validation(X_train, y_train, pipeline, parameter_cv, cv, grids[a])
        # path = './images/supervised/{}/train/with_resample/'.format(a)
        # evaluate_cross_validation(scores, confusion_matrixs, classification_reports, curves, path=path, show=False)

        # --------------------------- TRAIN MODEL WITH BEST PARAMS --------------------------- #

        path = './images/supervised/{}/best_resample/adasyn_'.format(a)
        cross_validation(X_train, y_train, pipeline, cv=cv, path=path)
        # model = find_best_estimator(X_train, y_train, pipeline, grids[a], parameter_cv, path=None)
        # tree_structure(model)
        # save_model(model, "decision_tree_model", data.shape[1], training_set=data, is_xgb=False)

        # ------------------------------------ TEST MODEL ------------------------------------ #

        # model_name = 'decision_tree_model'
        # y_pred_test = predict_profile(model_name, test_data)
        # path = './images/supervised/{}/best_resample/over_under/'.format(a)
        # y_pred_test = model.predict(X_test)
        # evaluate_predictions(y_test, y_pred_test, path=path, show=False)
