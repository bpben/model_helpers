## Classes for modelling using scikit-learn
This repo contains python classes that I find helpful for tuning and testing models.  They aren't exactly plug-and-play, but have some functionality that has been helpful in my work.

### Requirements
- Pandas
- Scikit-learn
- Numpy
- xgboost

### Indata class
Data container using Pandas DataFrames that can hold out a scoring set and split train/test sets according to specific criteria.  Splitter can also be prespecified for more flexibility.  Splitters must have `split` function.

Example usage:
```
X = np.zeros(shape=(3,4))
y = X[:,0]
d = Indata(X, y)
# default : ShuffleSplit
d.tr_te_split()
# specify splitter
sss = StratifiedShuffleSplit(**params)
d.tr_te_split(splitter=sss, y=y)
```

### Tuner class
Reads in Indata class that can be used to tune model hyperparameters.  

- Models currently implemented:
    - linear models (sklearn.linear\_model)
    - ensemble models (sklearn.ensemble)
    - xgboost models (xgboost)
    - SVM models (sklearn.svm)
- Requires a dictionary of parameters for the gridsearch (mparams) and for the cross-validation (cvparams)
- Can specify the CV method, though not all have been tested (default: K-fold)

The class will store all gridsearch results in a DataFrame and the best parameters in a dictionary keyed by the name of the model (user-provided)

Example usage:
```
#cv parameters
cvp = dict()
cvp['scoring'] = 'roc_auc'
cvp['n_iter'] = 3
cvp['iid'] = True
cv_method = StratifiedKFold
cv_method = cv_method(n_splits=5, shuffle=True)

#RF model parameters 
model = 'RandomForestClassifier'
mp = dict()
mp['n_estimators'] = ss.nbinom(n=2,p=0.01,loc=100)
mp['max_features'] = ss.beta(a=2,b=5)

d = Indata(X, y)
tune = Tuner(d)
tune.tune(model, features, mp, cv_method=cv_method, cvparams=cvp)
```

### Tester class
Using Indata class and an initiated model will train and test the model on the hold out set.  

- Can read in Tuner class and run best performing hyperparameters
- Default metrics:
    - Binary target: ROC AUC, F1_score, brier_score_loss
    - Target with more than 2 unique values: MAE, MSE, R^2
- Can specify metrics as a dictionary
- Can calibrate binary targets

Example usage:
```
d = Indata(X, y)
test = Tester(d)
# run tuned (from above)
test.init_tuned(tune)
te.run_model('RandomForestClassifier', tuned=True)
# run untuned
test.run_model('RandomForestClassifier', model=ske.RandomForestClassifier(), features=[0, 1])
```

