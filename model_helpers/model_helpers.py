import numpy as np
import pandas as pd
import sklearn.ensemble as ske
import sklearn.svm as svm
import sklearn.linear_model as skl
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, GroupKFold, GroupShuffleSplit, ShuffleSplit
from sklearn.calibration import CalibratedClassifierCV

class Indata(object):
    """
    Data format for feeding into Tuner/Tester classes
    """
    scoring = None
    data = None
    train_x, train_y, test_x, test_y = None, None, None, None
    is_split = 0
    
    def __init__(self, X, y, scoring=None):
        """
        X : all feature data, as pandas dataframe
        y : target data, as pd.Series or array
        scoring : idxs of scoring observations to not be included in training/test
        TODO: Not sure if we need this, actually
        """
        if scoring is not None:
            self.data = X[~(scoring)]
            self.scoring = X[scoring]
        else:
            self.data = X
        self.target = y
    
    def tr_te_split(self, pct=.7, splitter=None, seed=None, **kwargs):
        """
        Split into train/test
        pct : percent training observations, either this or a splitter need to be identified
        splitter : a sklearn.model_selection splitter, tested with Stratified Shuffle Split
        TODO: Test on others
        seed : random seed (optional)
        other arguments will be used with splitter
        """
        if splitter:
            self.splitter = splitter
        else:
            self.splitter = ShuffleSplit(n_splits=1, test_size=0.3)
        g = self.splitter.split(self.data, **kwargs)
        # get the actual indexes of the training set
        train, test = tuple(*g)
        if hasattr(self.data, 'index'):
            train = self.data.index.isin(self.data.index[train])
            test = ~train
        self.train_x = self.data[train]
        print('Train obs:', len(self.train_x))
        self.train_y = self.target[train]
        self.test_x = self.data[test]
        print('Test obs:', len(self.test_x))
        self.test_y = self.target[test]
        self.is_split = 1

class Tuner(object):
    """
    Initiates with indata class, will tune series of models according to parameters.  
    Outputs RandomizedGridCV results and parameterized model in dictionary
    """
    
    data = None
    train_x, train_y = None, None
    group_col = None
    
    def __init__(self, indata, best_models=None, grid_results=None):
        if indata.is_split == 0:
            raise ValueError('Data is not split, cannot be tested')
        # check if grouped by some column
        if hasattr(indata,'group_col'):
            self.group_col = indata.group_col
        self.data = indata.data
        self.train_x = indata.train_x
        self.train_y = indata.train_y
        if best_models is None:
            self.best_models = {}
        if grid_results is None:
            self.grid_results = pd.DataFrame()
        
            
    def make_grid(self, model, mparams, cv_method, cvparams):
        if cv_method is not None:
            cv = cv_method
        else:
            # default cv
            cv = KFold(n_splits=5, shuffle=True)
        grid = RandomizedSearchCV(
                    model(), cv=cv, param_distributions=mparams,
                    return_train_score=True,
                    **cvparams)
        return(grid)
    
    def run_grid(self, grid, train_x, train_y):
        grid.fit(train_x, train_y)
        results = pd.DataFrame(grid.cv_results_)[['mean_test_score','mean_train_score','params']]
        best = {}
        best['bp'] = grid.best_params_
        best[grid.scoring] = grid.best_score_
        return(best, results)
            
    def tune(self, m_name, features, mparams, cv_method=None, cvparams={}):
        """
        Randomized search for best parameters
        m_name : name of model class in sklearn
        features : feature names
        mparams : the parameters to use in grid search, keyed by the m_name
        cv_method : sklearn model selection class, initiated with parameters, KFold default
        cvparams : additional parameters to pass to randomized search
        """
        if hasattr(ske, m_name):
            model = getattr(ske, m_name)
        elif hasattr(skl, m_name):
            model = getattr(skl, m_name)
        elif hasattr(xgb, m_name):
            model = getattr(xgb, m_name)
        elif hasattr(svm, m_name):
            model = getattr(svm, m_name)
        else:
            raise ValueError('Model name is invalid.')
        grid = self.make_grid(model, mparams, cv_method, cvparams)
        best, results = self.run_grid(grid, self.train_x[features], self.train_y)
        results['name'] = m_name
        self.grid_results = self.grid_results.append(results)
        best['model'] = model(**best['bp'])
        best['features'] = list(features)
        self.best_models.update({m_name: best}) 
        
class Tester(object):
    """
    Initiates with indata class, receives parameterized sklearn models, prints and stores results
    """
    
    def __init__(self, data, rundict=None):
        if data.is_split == 0 :
            raise ValueError('Data is not split, cannot be tested')
        else:
            self.data = data
            if rundict is None:
                self.rundict = {}
            
    def init_tuned(self, tuned):
        """ pass Tuner object, populatest with names, models, features """
        if tuned.best_models=={}:
            raise ValueError('No tuned models found')
        else:
            self.rundict.update(tuned.best_models)
    
    def predsprobs(self, model, test_x):
        """ Produce predicted class and probabilities """
        # if the model doesn't have predict proba, will be treated as GLM
        if hasattr(model, 'predict_proba'):
            preds = model.predict(test_x)
            probs = model.predict_proba(test_x)
        else:
            probs = model.predict(test_x)
            preds = (probs>=.5).astype(int)
        return(preds, probs)
    
    def get_metrics(self, preds, probs, test_y, metric_dict={}):
        """ Produce metrics  
        TODO: Assumes metrics are applied to probabilities, not the right way to do this
        """
        result_metrics = {}
        if metric_dict=={}:
            if len(np.unique(test_y))==2:
                result_metrics['f1_s'] = metrics.f1_score(test_y, preds)
                result_metrics['roc'] = metrics.roc_auc_score(test_y, probs[:,1])
                result_metrics['brier'] = metrics.brier_score_loss(test_y, probs[:,1])            
            else:
                result_metrics['mae'] = metrics.mean_absolute_error(test_y, probs)
                result_metrics['r2'] = metrics.r2_score(test_y, probs)
                result_metrics['mse'] = metrics.mean_squared_error(test_y, probs)
        else:
            for metric in metric_dict:
                result_metrics[metric] = metric_dict[metric](test_y, probs)
        return(result_metrics)
    
    def make_result(self, model, test_x, test_y, metric_dict={}):
        """ gets predictions and runs metrics """
        preds, probs = self.predsprobs(model, test_x)
        result_metrics = self.get_metrics(preds, probs, test_y, metric_dict=metric_dict)
        for k in result_metrics:
            print('{}:{}'.format(k, result_metrics[k]))
        return(result_metrics)

    
    def run_model(self, name, model=None, features=None, 
                  cal=False, cal_m='sigmoid', tuned=False, metric_dict={}):
        """
        Fit and test model
        name (str) : name of model
        model (sklearn model object or None) :  model to fit and test
        features (list) : list of feature names
        cal (bool) : calibrate probabilities (TODO: Not fully tested)
        cal_m (str) : calibration method (TODO)
        tuned (bool) : whether to use the tuned parameters for the model
        metric_dict (dict) : dictionary of functions to use to score the model performance
        Will also store in rundict object
        """

        results = {}
        if name in self.rundict:
            if tuned:
                results['features'] = list(self.rundict[name]['features'])
                results['model'] = self.rundict[name]['model']
            else:
                # warn of overwrite
                print('overwriting previous %s results' % name)
                results['features'] = list(features)
                results['model'] = model
        else:
            results['features'] = list(features)
            results['model'] = model
        print("Fitting {} model with {} features".format(name, results['features']))
        if cal:
            # Need disjoint calibration/training datasets
            # Split 50/50
            rnd_ind = np.random.rand(len(self.data.train_x)) < .5
            train_x = self.data.train_x[results['features']][rnd_ind]
            train_y = self.data.train_y[rnd_ind]
            cal_x = self.data.train_x[results['features']][~rnd_ind]
            cal_y = self.data.train_y[~rnd_ind]
        else:
            train_x = self.data.train_x[results['features']]
            train_y = self.data.train_y

        self.m_fit = results['model'].fit(train_x, train_y)
        m_fit = results['model'].fit(train_x, train_y)
        result = self.make_result(
            m_fit,
            self.data.test_x[results['features']],
            self.data.test_y,
            metric_dict=metric_dict)

        results['raw'] = result
        results['m_fit'] = m_fit
        if cal:
            print("calibrated:")
            m_c = CalibratedClassifierCV(results['model'], method = cal_m)
            m_fit_c = m_c.fit(cal_x, cal_y)
            result_c = self.make_result(m_fit_c, 
                                        self.data.test_x[results['features']], 
                                        self.data.test_y)
            results['calibrated'] = result_c              
            print("\n")
        if name in self.rundict:
            self.rundict[name].update(results)
        else:
            self.rundict.update({name:results})