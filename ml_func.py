import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

def analyze_accuracy(y_test, y_pred_prob, y_pred_class):
    naive_pred = [1 for i in range(len(y_pred_class))]
    
    ns_auc = roc_auc_score(y_test, naive_pred)
    md_auc = roc_auc_score(y_test, y_pred_prob)
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Model   : ROC AUC=%.3f' % (md_auc))
    
    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(cm, annot=True)
    plt.show()
    
    ns_fpr, ns_tpr, _ = roc_curve(y_test, naive_pred)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred_prob)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
def get_predictions(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred_class = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)
    y_pred_prob = [elem[1] for elem in y_pred_prob]
    
    return y_pred_class, y_pred_prob
    
def plot_calibration(y_test, y_prob):
    fop, mpv = calibration_curve(y_test,y_prob, n_bins=10, normalize=True)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(mpv, fop, marker='.')
    plt.show()

def setup_chess_data():
    df = pd.read_csv("chess_king_rook_dataset.csv")
    
    df['win'] = df['result'] != 'draw'
    df['win'] = df['win'].astype(int)
    
    for col in df.columns:
        if "rank" in col:
            df[col] = df[col].astype(str)

    dummy_df = copy.deepcopy(df)
    del dummy_df['result']
    
    dummy_df = pd.get_dummies(dummy_df)
    for col in dummy_df.columns:
        dummy_df[col] = dummy_df[col].astype(float)
    dummy_df['result'] = df['result']
    
    col_list = []
    for col in dummy_df.columns:
        if col != 'win' and col != 'result':
            col_list.append(col)

    X = dummy_df.loc[:, col_list]
    y = dummy_df['win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
    return X_train, X_test, y_train, y_test