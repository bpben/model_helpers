## Classes for modelling using scikit-learn
This repo contains python classes that I find helpful for tuning and testing models.  They aren't exactly plug-and-play, but have some functionality that has been helpful in my work.

### Requirements
- Pandas
- Scikit-learn
- Numpy
- xgboost

### Indata class
Data container using Pandas DataFrames that can hold out a scoring set and split train/test sets according to specific criteria.  If you'd prefer date-sorted train/test (e.g. for forecasting), you can specify a date column.

### Tuner class
Reads in Indata class that can be used to tune model hyperparameters.  

Notes: 
- Relies on RandomizedGridSearch with K-fold crossvalidation
- Currently only implements linear models (sklearn.linear\_model) or ensemble models (sklearn.ensemble)
    - The Testing class is designed for classifier models and will return an error for continuous targets
- Requires a dictionary of parameters for the gridsearch (mparams) and for the cross-validation (cvparams)

The class will store all gridsearch results in a DataFrame and the best parameters in a dictionary keyed by the name of the model (user-provided)

### Tester class
Using Indata class and an initiated model will train and test the model.  It will return metrics for both the calibrated and uncalibrated model.  Can also read in a Tuner class and run models using the tuned hyperparameters.  Is able to write the results of the models to csv.
