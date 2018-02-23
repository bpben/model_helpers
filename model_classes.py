import numpy as np
import pandas as pd
import sklearn.ensemble as ske
import sklearn.svm as svm
import sklearn.linear_model as skl
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, KFold, StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

class Indata():
    scoring = None
    data = None
    train_x, train_y, test_x, test_y = None, None, None, None
    is_split = 0
    
    #init with pandas DF and target column name, specify scoring observations
    def __init__(self, data, target, scoring=None):
        #If scoring observations, store under scoring attribute
        if scoring is not None:
            self.data = data[~(scoring)]
            self.scoring = data[scoring]
        else:
            self.data = data
        self.target = target
    
    # Split into train/test
    # pct : percent training observations
    # datesort : specify date column for sorting values
    #   If this is not None, split will be non-random (i.e. split on sorted obs)
    def tr_te_split(self, pct, datesort=None, group_col=None, seed=None):
        """
        Split into train/test
        pct : percent training observations
        datesort : specify date column for sorting values
            If this is not None, split will be non-random (i.e. split on sorted obs)
        group_col : group column name for groupkfold split
            Will also be passed to tuner
        """
        if group_col:
            self.group_col = group_col
            grouper = GroupShuffleSplit(n_splits=1, train_size=pct)
            g = grouper.split(self.data, groups=self.data[group_col])
            # get the actual indexes of the training set
            inds, _ = tuple(*g)
            # translate that into boolean array
            inds = self.data.index[inds]
            inds = self.data.index.isin(inds)
        elif datesort:
            self.data.sort_values(datesort, inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            inds = np.arange(0.0,len(self.data)) / len(self.data) < pct
        else:
            np.random.seed(seed)
            inds = np.random.rand(len(self.data)) < pct
        self.train_x = self.data[inds]
        print 'Train obs:', len(self.train_x)
        self.train_y = self.data[self.target][inds]
        self.test_x = self.data[~inds]
        print 'Test obs:', len(self.test_x)
        self.test_y = self.data[self.target][~inds]
        self.is_split = 1
        
class Tuner():
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
        
            
    def make_grid(self, model, cvparams, mparams):
        #Makes CV grid
        # to implement, no capability for GroupKFold for randomizedsearch
        #if self.group_col:
            #cv = GroupKFold(cvparams['folds'])
        grid = RandomizedSearchCV(
                    model(),scoring=cvparams['pmetric'], 
                    cv = KFold(cvparams['folds'], cvparams['shuffle']),
                    refit=False, n_iter=cvparams['iter'],
                    param_distributions=mparams, verbose=1)
        return(grid)
    
    def run_grid(self, grid, train_x, train_y):
        grid.fit(train_x, train_y)
        results = pd.DataFrame(grid.cv_results_)[['mean_test_score','mean_train_score','params']]
        best = {}
        best['bp'] = grid.best_params_
        best[grid.scoring] = grid.best_score_
        return(best, results)
            
    def tune(self, name, m_name, features, cvparams, mparams):
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
        grid = self.make_grid(model, cvparams, mparams)
        best, results = self.run_grid(grid, self.train_x[features], self.train_y)
        results['name'] = name
        results['m_name'] = m_name
        self.grid_results = self.grid_results.append(results)
        best['model'] = model(**best['bp'])
        best['features'] = list(features)
        self.best_models.update({name: best}) 
        
class Tester():
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
            probs = model.predict_proba(test_x)[:,1]
        else:
            probs = model.predict(test_x)
            preds = (probs>=.5).astype(int)
        return(preds, probs)
    
    def get_metrics(self, preds, probs, test_y):
        """ Produce metrics  """
        result_metrics = {}
        if len(np.unique(test_y))==2:
            result_metrics['f1_s'] = metrics.f1_score(test_y, preds)
            result_metrics['roc'] = metrics.roc_auc_score(test_y, probs)
            result_metrics['brier'] = metrics.brier_score_loss(test_y, probs)            
        else:
            result_metrics['mae'] = metrics.mean_absolute_error(test_y, probs)
            result_metrics['r2'] = metrics.r2_score(test_y, probs)
            result_metrics['mse'] = metrics.mean_squared_error(test_y, probs)
        return(result_metrics)
    
    def make_result(self, model, test_x, test_y):
        """ gets predictions and runs metrics """
        preds, probs = self.predsprobs(model, test_x)
        result_metrics = self.get_metrics(preds, probs, test_y)
        for k in result_metrics:
            print '{}:{}'.format(k, result_metrics[k])
        return(result_metrics)

    
    def run_model(self, name, model, features, cal=True, cal_m='sigmoid'):
        """
        Run a specific model (not from Tuner classs)
        By default, calibrates predictions and produces metrics for them
        Will also store in rundict object
        """

        results = {}
        results['features'] = list(features)
        results['model'] = model
        print "Fitting {} model with {} features".format(name, len(features))
        if cal:
            # Need disjoint calibration/training datasets
            # Split 50/50
            rnd_ind = np.random.rand(len(self.data.train_x)) < .5
            train_x = self.data.train_x[features][rnd_ind]
            train_y = self.data.train_y[rnd_ind]
            cal_x = self.data.train_x[features][~rnd_ind]
            cal_y = self.data.train_y[~rnd_ind]
        else:
            train_x = self.data.train_x[features]
            train_y = self.data.train_y

        m_fit = model.fit(train_x, train_y)
        result = self.make_result(
            m_fit,
            self.data.test_x[features],
            self.data.test_y)

        results['raw'] = result
        results['m_fit'] = m_fit
        if cal:
            print "calibrated:"
            m_c = CalibratedClassifierCV(model, method = cal_m)
            m_fit_c = m_c.fit(cal_x, cal_y)
            result_c = self.make_result(m_fit_c, self.data.test_x[features], self.data.test_y)
            results['calibrated'] = result_c              
            print "\n"
        if name in self.rundict:
            self.rundict[name].update(results)
        else:
            self.rundict.update({name:results})
    
    def run_tuned(self, name, cal=True, cal_m='sigmoid'):
        """ Wrapper for run_model when using Tuner object """
        self.run_model(name, self.rundict[name]['model'], self.rundict[name]['features'], cal, cal_m)

    def lift_chart(self, x_col, y_col, data, ax=None, pct=True):
        """ 
        create lift chart 
        x_col = pctiles of predictions
        y_col = % positive class
        """
        p = sns.barplot(x=x_col, y=y_col, data=data, 
                        palette='Greens', ax = None, ci=None)
        vals = p.get_yticks()
        xvals = [x.get_text().split(',')[-1].strip(']') for x in p.get_xticklabels()]
        if pct==True:
            p.set_yticklabels(['{:3.0f}%'.format(i*100) for i in vals])
            xvals = ['{:2.1f}%'.format(float(x)*100) for x in xvals]
        p.set_xticklabels(xvals, rotation=30)
        p.set_facecolor('white')
        p.set_xlabel('')
        p.set_ylabel('')
        p.set_title('Predicted probability vs actual percent')
        return(p)
    
    def density(self, data, score_col, ax=None):
        """ create kdeplot of predictions """
        p = sns.kdeplot(data[score_col], ax=ax)
        p.set_facecolor('white')
        p.legend('')
        p.set_xlabel('Predicted probability')
        p.set_title('KDE plot predictions')
        return(p)

    def density_and_lift_charts(self, model, features=None, model_params=None, verbose=True, qcut=10):
        """ 
        produces prediction density and decile lift chart 
        currently only works for binary targets (0/1)
        model (str or object with predict) : name in rundict (if used), otherwise model
        features (list) : list of features, if not available in rundict
        model_params : can just pass model params (from rundict)
        verbose : True if you want the prediction deciles to be output
        qcut : can specify percentile cut (default = decile)
        """
        if model_params:
            pass
        elif model not in self.rundict:
            preds, probs = self.predsprobs(model, self.data.test_x[features])
        else:
            model_params = self.rundict[model]
            preds, probs = self.predsprobs(model_params['m_fit'],
                self.data.test_x[model_params['features']])
        risk_df = pd.DataFrame(
            {'probs':probs, 'target':self.data.test_y})
        risk_df['categories'] = pd.qcut(risk_df['probs'], qcut)
        risk_mean = risk_df.groupby('categories')['target'].mean().reset_index()
        if verbose:
            print risk_df.probs.describe()
        fig, axes = plt.subplots(1, 2)
        self.lift_chart('categories', 'target', risk_df, 
                   ax=axes[1])
        self.density(risk_df, 'probs', ax=axes[0])
        return(risk_df)

    
    def to_csv(self):
        """ outputs rundict to csv """
        if self.rundict == {}:
            raise ValueError('No results found')
        else:
            now = pd.to_datetime('today').value
            #Make dataframe, transpose so each row = model
            pd.DataFrame(self.rundict).T.to_csv('results_{}.csv'.format(now))
