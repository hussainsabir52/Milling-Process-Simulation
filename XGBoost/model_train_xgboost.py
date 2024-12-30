import xgboost as xgb
import numpy as np

X_train = np.loadtxt("X_train.txt")
X_test = np.loadtxt("X_test.txt")

y_train = np.loadtxt("y_train.txt")
y_test = np.loadtxt("y_test.txt")

y_train=y_train[:,:1] #dx
y_test=y_test[:,:1] #dx

dtrain=xgb.DMatrix(X_train, label=y_train.flatten())
dtest=xgb.DMatrix(X_test, label=y_test.flatten())



param = {'eta':0.8,'max_depth':5}
param['nthread'] = 4

evallist = [(dtrain, 'train'), (dtest, 'eval')]

num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)

bst.load_model('xgboost_gbtree.model')

ypred = bst.predict(dtest)

np.savetxt("y_pred.txt", ypred)