from sklearn.metrics import accuracy_score, precision_score, roc_auc_score


def scores(y_test, y_pred):
    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), roc_auc_score(y_test, y_pred)
        