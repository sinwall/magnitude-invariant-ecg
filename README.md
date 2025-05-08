# Magnitude theory and Electrocardiogram

<h1>Preparation</h1>

- Python == 3.8.6
- packages as in `requirements.txt`
- databases: FANTASIA, NSRDB, MITDB, AFDB: these are all available from PhysioNet. 
The code assumes these datasets are unzipped at `../data/`. 

<h1>How to run</h1>
Please execute the following script in terminal:

```
> python main.py
```

then experiment will begin and scores will be printed in console and output log file.

Note: cross validation result is saved as a csv file(`.validated.csv`). 

By default, the file will be used rather than do cross validation again, to save time. 

<h1>Output format</h1>

After run, .csv file `.performed.csv` is generated. 
Each field refers to:

- `dbname`: name of the dataset
- `weight_type`: weighting, diversifier, uniform or default(fourier transform not used)
- `model_name`: lr(logistic regression), knn(K-nearest neighbors), svm(support vector machine), mlp(multi-layer perceptron)
- `acc_{mean,std}`: mean accuracy or standard deviation
- `calc_time_{fit,pred}`: calculation time for fitting(training) and predicting
- `calc_time_coef`: calculation time for weighting or diversifier
- `calc_time_dist`: calculation time for distant feature
- `calc_time_fourier`: calculation time for fourier transform