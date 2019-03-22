import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

def model_selection(X_train,X_test,y_train,y_test):
    """Optimizes each model and selects most accurate using grid search cv"""
    scaler=StandardScaler()
    pipe_logreg = Pipeline([('scl',scaler),('clf',LogisticRegression(random_state=42))])
    pipe_net = Pipeline([('scl',scaler),('clf',MLPClassifier())])
    pipe_xgboost = Pipeline([('scl',scaler),('clf',XGBClassifier())])

    # Set grid search parameters
    jobs = -1
    l_rate_init = [0.001]
    momentum = [0.0,0.2,0.4,0.6,0.8,0.9]
    iterate = [2000]
    alph = [.0001,.001,.01,.1]
    random_state = [0,1,2,3,4,5,6,7,8,9]
    gam = [.5,1,1.5,2]
    max_depth = [3,4,5]
    min_child_weight = [1,3,5]
    sub_sample = [.6,.8,1.0]
    colsample = [.6,.8,1.0]
    reg_strength = [.01,.5,1.0,1.5,2.0]
    layers = [(50,100,50),(100,)]
    learn_rate = ["constant"]
    log_pen = ['l2']
    activate = ['logistic','relu','tanh','identity']
    net_solvers = ['adam']
    log_solver = ['liblinear','lbfgs','saga','newton-cg']
    penalty,c,solver,alpha,activation = 'clf__penalty','clf__C','clf__solver','clf__alpha','clf__activation'
    colsamp,iterations,mome,lrate = 'clf__colsample_bytree','clf__max_iter','clf__momentum','clf__learning_rate'
    rs,hidden_layers,learn_rate_init = 'clf__random_state','clf__hidden_layer_sizes','clf__learning_rate_init'
    gamma,maxdepth,minchild,subsamp = 'clf__gamma','clf__max_depth','clf__min_child_weight','clf__subsample'

    # Logistic Regression Parameters
    logreg_params = [{penalty:log_pen,
                      rs:random_state,
                      solver:log_solver,
                      c:reg_strength}]

    # Multilayer Perceptron Parameters
    net_params = [{solver:net_solvers,
                   activation:activate,
                   alpha:alph,
                   hidden_layers:layers,
                   learn_rate_init:l_rate_init,
                   mome:momentum,
                   lrate:learn_rate,
                   iterations:iterate}]

    # XGboost Parameters
    xgboost_params = {gamma:gam,
                      maxdepth:max_depth,
                      minchild:min_child_weight,
                      subsamp:sub_sample,
                      colsamp:colsample}

    # Construct grid searches
    gs_logreg = GridSearchCV(estimator=pipe_logreg,param_grid=logreg_params,scoring='accuracy',cv=10,n_jobs=jobs)
    gs_net = GridSearchCV(estimator=pipe_net,param_grid=net_params,scoring='accuracy',cv=10,n_jobs=jobs)
    gs_xgboost = GridSearchCV(estimator=pipe_xgboost,param_grid=xgboost_params,scoring='accuracy',cv=10,n_jobs=jobs)

    # Begin model training and comparison
    print('Performing model optimizations...')
    grids = [gs_logreg,gs_net,gs_xgboost]
    grid_dict = {0: 'Logistic Regression', 1: 'Multilayer Perceptron', 2: 'XGboost'}
    best_acc = 0.0
    best_clf = 0
    best_gs = ''
    preds = []
    for idx, gs in enumerate(grids):
        print('\nModel: %s' % grid_dict[idx])
        gs.fit(X_train,y_train)
        print('Best parameters: %s' % gs.best_params_)
        # Best training data accuracy
        print('Best training accuracy: %.3f' % gs.best_score_)
        # Predict on test data with best parameters
        y_pred = gs.predict(X_test)
        preds.append(y_pred)
        # Test data accuracy of model with best params
        print('Test set accuracy score for best parameters: %.3f ' % accuracy_score(y_test, y_pred))
        # Track best (highest test accuracy) model
        if accuracy_score(y_test,y_pred)>best_acc:
            best_acc = accuracy_score(y_test,y_pred)
            best_gs = gs
            best_clf = idx
    print('\nClassifier with best test set accuracy: %s' % grid_dict[best_clf])
    return preds

def roc_generator(y_test,pred1,pred3,pred4,):
    """Roc curve comparing each model's performance"""
    num_classes = 2

    # Logistic regression true/false positive
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Multilayer Perceptron true/false positive
    fpr2 = dict()
    tpr2 = dict()
    roc_auc2 = dict()

    # Xgboost true/false positive
    fpr3 = dict()
    tpr3 = dict()
    roc_auc3 = dict()

    for i in range(num_classes):
        fpr[i],tpr[i],_ = roc_curve(y_test,pred1,pos_label=1)
        roc_auc[i] = auc(fpr[i],tpr[i])
    for i in range(num_classes):
        fpr2[i],tpr2[i],_ = roc_curve(y_test,pred3,pos_label=1)
        roc_auc2[i] = auc(fpr2[i],tpr2[i])
    for i in range(num_classes):
        fpr3[i],tpr3[i], _ = roc_curve(y_test,pred4,pos_label=1)
        roc_auc3[i] = auc(fpr3[i],tpr3[i])
    fig = plt.figure(figsize=(15,10),dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.set_facecolor('#c1c5cc')
    major_ticks = np.arange(0.0,1.0,0.05)
    minor_ticks = np.arange(0.0,1.0,0.05)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks,minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks,minor=True)
    ax.grid(which='both')

    # Logistic Regression
    plt.plot(fpr[1],tpr[1],color='#4a50bf',
    lw=1,label='Logistic Regression (area = %0.4f)' % roc_auc[1])
    plt.plot([0,1],[0,1],color='black',lw=1,linestyle='--')

    # Multilayer Perceptron
    plt.plot(fpr3[1],tpr3[1],color='#F6FF33',lw=1,
    label='Multilayer Perceptron (area = %0.4f)' % roc_auc3[1])
    plt.plot([0,1],[0,1],color='black',lw=1,linestyle='--')

    # Xgboost
    plt.plot(fpr3[1],tpr3[1],color='#ff68f0',lw=1,
    label='Xgboost (area = %0.4f)' % roc_auc3[1])
    plt.plot([0,1],[0,1],color='black',lw=1,linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristics Curve')
    plt.legend(loc="lower right")
    fig1 = plt.gcf()
    plt.draw()
    fig1.savefig('roc.png',dpi=100)

def main():
    diabetes = pd.read_csv('/Users/nickalonso/Downloads/diabetes.csv')
    dataframe = pd.DataFrame(diabetes)
    X = dataframe.loc[:,diabetes.columns != 'Outcome']
    y = dataframe['Outcome']
    X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y,random_state=66)
    predictions = model_selection(X_train,X_test,y_train,y_test)
    pred1,pred3,pred4 = predictions[0],predictions[1],predictions[2]

    # Generate ROC curve for model performance comparision
    roc_generator(y_test,pred1,pred3,pred4)

if __name__ == '__main__':
    main()
