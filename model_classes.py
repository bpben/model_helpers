import numpy as np
import pandas as pd
import sklearn.ensemble as ske
import sklearn.linear_model as skl
from sklearn import metrics
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn import cross_validation as cv

class indata():
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
    # pct = percent training observations
    # datesort = specify date column for sorting values
    #   If this is not None, split will be non-random (i.e. split on sorted obs)
    def tr_te_split(self, pct, datesort=None):
        if datesort:
            self.data.sort_values(datesort, inplace=True)
            inds = np.arange(0.0,len(self.data)) / len(self.data) < pct
        else:
            inds = np.random.rand(len(self.data)) < pct
        self.train_x = self.data[inds]
        print 'Train obs:', len(self.train_x)
        self.train_y = self.data[self.target][inds]
        self.test_x = self.data[~inds]
        print 'Test obs:', len(self.test_x)
        self.test_y = self.data[self.target][~inds]
        self.is_split = 1
        
class tuner(indata):
    """
    Initiates with indata class, will tune series of models according to parameters.  
    Outputs RandomizedGridCV results and parameterized model in dictionary
    """
    
    grid_results = pd.DataFrame()
    best_models = {}
    data = None
    train_x, train_y = None, None
    
    def __init__(self, indata):
        if indata.is_split == 0 :
            raise ValueError('Data is not split, cannot be tested')
        else:
            self.data = indata.data
            self.train_x = indata.train_x
            self.train_y = indata.train_y
        
            
    def make_grid(self, model, obs, cvparams, mparams):
        #Makes CV grid
        grid = RandomizedSearchCV(
                    model(),scoring=cvparams['pmetric'], 
                    cv = cv.KFold(obs,cvparams['folds']), 
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
        else:
            raise ValueError('Model name is invalid.')
        grid = self.make_grid(model, len(self.train_x), cvparams, mparams)
        best, results = self.run_grid(grid, self.train_x[features], self.train_y)
        results['name'] = name
        results['m_name'] = m_name
        self.grid_results = self.grid_results.append(results)
        best['model'] = model(**best['bp'])
        best['features'] = list(features)
        self.best_models.update({name: best}) 
        
class tester():
    """
    Initiates with indata class, receives parameterized sklearn models, prints and stores results
    """
    rundict = {}
    data = None
    
    def __init__(self, data):
        if data.is_split == 0 :
            raise ValueError('Data is not split, cannot be tested')
        else:
            self.data = data
            
    #Add tuner object, will populate rundict with names, models, feature
    def init_tuned(self, tuned):
        if tuned.best_models=={}:
            raise ValueError('No tuned models found')
        else:
            self.rundict.update(tuned.best_models)
    
    #Produce predicted class and probabilities
    def predsprobs(self, model, test_x):
        preds = model.predict(test_x)
        probs = model.predict_proba(test_x)[:,1]
        return(preds, probs)
    
    #Produce metrics
    def get_metrics(self, preds, probs, test_y):
        f1_s = metrics.f1_score(test_y, preds)
        brier = metrics.brier_score_loss(test_y, probs)
        return(f1_s, brier)
    
    #Run production, output dictionary
    def make_result(self, model, test_x, test_y):
        preds, probs = self.predsprobs(model, test_x)
        f1_s, brier = self.get_metrics(preds, probs, test_y)
        print "f1_score: ", f1_s
        print "brier_score: ", brier
        result = {}
        #result['preds'] = [int(i) for i in preds]
        #result['probs'] = [float(i) for i in probs]
        result['f1_s'] = f1_s
        result['brier'] = brier
        return(result)
    
    #Run model - Specify model, with parameters, features
    #Stores it to rundict, can later be output
    #Will overwrite previous run if name is not different
    def run_model(self, name, model, features, cal=True, cal_m='sigmoid'):
        results = {}
        results['features'] = list(features)
        print "Fitting {} model with {} features".format(name, len(features))
        
        m_fit = model.fit(self.data.train_x[features], self.data.train_y)
        result = self.make_result(m_fit, self.data.test_x[features], self.data.test_y)
        results['raw'] = result
        results['m_fit'] = m_fit
       
        if cal==True:
            print "calibrated:"
            m_c = CalibratedClassifierCV(m_fit, method = cal_m, cv='prefit')
            m_fit_c = m_c.fit(self.data.train_x[features], self.data.train_y)
            result_c = self.make_result(m_fit_c, self.data.test_x[features], self.data.test_y)
            results['calibrated'] = result_c              
            print "\n"
        if name in self.rundict:
            self.rundict[name].update(results)
        else:
            self.rundict.update({name:results})
    
    #Run from tuned set
    def run_tuned(self, name, cal=True, cal_m='sigmoid'):
        self.run_model(name, self.rundict[name]['model'], self.rundict[name]['features'], cal, cal_m)
    
    #Output rundict to csv
    def to_csv(self):
        if self.rundict == {}:
            raise ValueError('No results found')
        else:
            now = pd.to_datetime('today').value
            #Make dataframe, transpose so each row = model
            pd.DataFrame(self.rundict).T.to_csv('results_{}.csv'.format(now))