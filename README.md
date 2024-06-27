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

After run, .txt files with names `.summary-{dbname}.txt` are generated. 
Each file contains statistics of scores, calculation time, etc. The line starts with
`statistics of {model_name}-{experiment_type}` records mean, std, max, min of accuracy scores. 
Here, `model_name` is one of `lr`, `knn`, `svm`, `mlp`. `experiment_type` is one of `weighting`, `diversifier`, `uniform`, `default`.