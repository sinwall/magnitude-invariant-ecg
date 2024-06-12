import itertools
from collections import defaultdict
from datetime import datetime
import logging
# import os
# import sys
# sys.path.append(os.path.abspath('../'))
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tools import (
    load_data, remove_baseline, resample_ecg, divide_segments,
    make_curves, compress_curves, calculate_weights, calculate_max_dispers,
    extract_fourier, extract_distance,
    split_into_train_test, save_features,
    calculate_weighting_vectors,
)


# logger_file = logging.getLogger(__name__)
# logger_file.addHandler(logging.FileHandler('.log.txt'))
logger_stream = logging.getLogger(__name__)
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger_stream.addHandler(logger_stream_handler)
logger_stream.setLevel(logging.INFO)
logging.basicConfig(
    filename='.log.txt',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)


def write_log(msg):
    # logging.info(msg)
    logger_stream.info(msg)

def compose(*args):
    def _inner(x):
        for fn in args:
            x = fn(x)
        return x
    return _inner


def load_validated_params(file_name, dbname, split_no, weight_type='w'):
    df = pd.read_csv(file_name, na_values=['-'])
    df = df[df['dbname'] == dbname]
    df = df[df['split_no'] == split_no]
    df = df[df['weight_type'] == str(weight_type)]
    df = df.iloc[0]

    scale_w = df['scale_w']
    scale_d = df['scale_d']
    list_of_best_params = {
        'lr': dict(C=df['LR-C']),
        'knn': dict(n_neighbors=df['KNN-n_neigh'], weights=df['KNN-weight']),
        'svm': dict(C=df['SVM-C'], gamma=df['SVM-gamm']),
        'mlp': dict(hidden_layer_sizes=(df['MLP-hidden'], ))
    }
    return scale_w, scale_d, list_of_best_params


def get_model(name, random_state=None, **kwargs):
    if name == 'knn':
        kwargs_default = dict(
            # n_neighbors=2,
            # weights='distance',
        )
        kwargs_default.update(kwargs)
        kwargs = kwargs_default
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                **kwargs
            )
        )
    elif name == 'hgbt':
        kwargs_default = dict(
            # max_iter=100,
            # max_depth=10,
            random_state=random_state
        )
        kwargs_default.update(kwargs)
        kwargs = kwargs_default
        model = HistGradientBoostingClassifier(
            **kwargs
        )
    elif name == 'lr':
        kwargs_default = dict(
            # C=1e0,
            max_iter=10000,
            random_state=random_state
        )
        kwargs_default.update(kwargs)
        kwargs = kwargs_default
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                **kwargs
            )
        )
    elif name == 'svm':
        kwargs_default = dict(
            random_state=random_state
        )
        kwargs_default.update(kwargs)
        kwargs = kwargs_default
        model = make_pipeline(
            StandardScaler(),
            SVC(**kwargs)
        )
    elif name == 'mlp':
        kwargs_default = dict(
            random_state=random_state
        )
        kwargs_default.update(kwargs)
        kwargs = kwargs_default
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(**kwargs)
        )
    return model


def cache_segments(dbname, dbpath, random_states=(42, )):
    data_bundle = dict()
    data_bundle = compose(
        load_data(dbname, dbpath),
        remove_baseline(),
        resample_ecg(fs_after=250),
    )(data_bundle)
    for random_state in random_states:
        data_bundle = compose(
            divide_segments(seg_dur=2, fs=250, minmax_scale=False, random_state=random_state, segs_per_person=1000),
        )(data_bundle)
        np.savez(
            f'.cache/segs-{dbname}-random_state={random_state}', 
            segs=data_bundle['segs'],
            seg_ids=data_bundle['seg_ids']
        )

def load_cached_segments(data_bundle, dbname, random_state):
    npz_loaded = np.load(f'.cache/segs-{dbname}-random_state={random_state}.npz')
    for key, val in npz_loaded.items():
        data_bundle[key] = val
    return data_bundle


def select_geometric_params(
        dbname, dbpath, dim=3, lag=5, scale_ws=tuple(), scale_ds=tuple(), weight_type=False, 
        downsample_size=100, distance_scale=1e0, n_filters_small=256, n_filters=1024,
        model_names=[], param_grids=[],
        split_no=42, use_cache=False, 
        ):
    '''
    This function is used to choose best `lag` and radii(`scale_w`, `scale_d`). 
    '''
    # prepare and preprocess data
    # and save into `data_bundle`
    data_bundle = dict()
    if use_cache:
        data_bundle = load_cached_segments(data_bundle, dbname, split_no)
    elif not use_cache:
        data_bundle = compose(
            load_data(dbname, dbpath),
            remove_baseline(),
            resample_ecg(fs_after=250),
                divide_segments(seg_dur=2, fs=250, minmax_scale=False, random_state=split_no, segs_per_person=1000),
        )(data_bundle)
    data_bundle = compose(
        make_curves(dim=dim, lag=lag, reduce=0),  # time-delay embedding
        compress_curves(size=downsample_size),  # reduce
    )(data_bundle)
    # clock = datetime.now()
    if weight_type == 'w':
        data_bundle = calculate_weights(scale=distance_scale)(data_bundle)
    elif weight_type == 'm':
        data_bundle = calculate_max_dispers(scale=distance_scale)(data_bundle)
    # stats[f'calc-time-{weight_type}'].append( (datetime.now()-clock).total_seconds() )
        
    y = data_bundle['seg_ids']
    idwise_cnt_normalized = np.sum([
        (y == s)*np.cumsum((y == s) / np.sum(y == s))
        for s in np.unique(y)
    ], axis=0)
    mask_train = (idwise_cnt_normalized < 0.8)
    mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized >=0.6)
    # mask_test = (idwise_cnt_normalized > 0.8)

    model = get_model('knn')
    stats = defaultdict(list)
    for random_state in range(42, 42+5):
        for scale_d in scale_ds:
            data_bundle = compose(
                extract_distance(scale=scale_d, n_filters=n_filters_small, random_state=random_state),
            )(data_bundle)
            X_dist = data_bundle['X_dist']

            for scale_w in scale_ws:
                if weight_type:
                    # clock = datetime.now()
                    data_bundle = extract_fourier(scale=scale_w, n_filters=n_filters_small//2, weight_type=weight_type, random_state=random_state)(data_bundle)
                    # stats[f'calc-time-fourier-{weight_type}'].append( (datetime.now()-clock).total_seconds() )
                    X_fourier = data_bundle[f'X_fourier_{weight_type}']
                    X = np.concatenate([X_dist[..., ::2], X_fourier], axis=1)
                else:
                    X = X_dist
    
                y_train, y_test = y[mask_train&~mask_val], y[mask_val]
                X_train, X_test = X[mask_train&~mask_val], X[mask_val]

                # clock = datetime.now()
                model.fit(X_train, y_train)
                # stats['calc-time-fit'].append( (datetime.now()-clock).total_seconds() )

                # clock = datetime.now()
                score = accuracy_score(y_test, model.predict(X_test))
                # stats['calc-time-pred'].append( (datetime.now()-clock).total_seconds() )

                write_log(score)
                stats[f'score-{scale_w}-{scale_d}'].append(score)
    params_max = None
    score_max = 0
    write_log('result:')
    for scale_w, scale_d in itertools.product(scale_ws, scale_ds):
        scores = stats[f'score-{scale_w}-{scale_d}']
        write_log(f'{scale_w}-{scale_d}: mean={np.mean(scores)}, std={np.std(scores)}')
        if np.mean(scores) > score_max:
            params_max = (scale_w, scale_d)
            score_max = np.mean(scores)
    write_log(f'geom-params for {dbname}-{split_no}: params_max={params_max}, score_max={score_max}')
    write_log('')
    scale_w, scale_d = params_max
    
    stats = defaultdict(list)
    for random_state in range(42, 42+5):
        data_bundle = compose(
            extract_distance(scale=scale_d, n_filters=n_filters, random_state=random_state),
        )(data_bundle)
        X_dist = data_bundle['X_dist']
        if weight_type:
            data_bundle = extract_fourier(scale=scale_w, n_filters=n_filters//2, weight_type=weight_type, random_state=random_state)(data_bundle)
            X_fourier = data_bundle[f'X_fourier_{weight_type}']
            X = np.concatenate([X_dist[..., ::2], X_fourier], axis=1)
        else:
            X = X_dist
        y_train, y_test = y[mask_train&~mask_val], y[mask_val]
        X_train, X_test = X[mask_train&~mask_val], X[mask_val]

        for model_name, param_grid in zip(model_names, param_grids):
            write_log(model_name)
            param_names = param_grid.keys()
            for param_vals in itertools.product(*[param_grid[name] for name in param_names]):
                params = {name: val for name, val in zip(param_names, param_vals)}
                write_log(str(params))
                model = get_model(model_name, random_state=random_state, **params)
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
                stats[(model_name,) + tuple(params.items())].append( score )
                write_log(score)
            write_log('')
        write_log('')

    list_of_best_params = dict()
    for model_name, param_grid in zip(model_names, param_grids):
        write_log(f'result: {model_name}')
        param_names = param_grid.keys()
        params_max = None
        score_max = 0
        for param_vals in itertools.product(*[param_grid[name] for name in param_names]):
            params = {name: val for name, val in zip(param_names, param_vals)}
            score = np.mean(stats[(model_name,) + tuple(params.items())])
            write_log(f'{model_name}, {params}, score={score}')
            if score > score_max:
                score_max = score
                params_max = params
        list_of_best_params[model_name] = params_max
        write_log(f'model params for {dbname}-{split_no}, {model_name}: params={params_max} with score {score_max}')
    return scale_w, scale_d, list_of_best_params
    

# def select_model_params(
#         dbname, dbpath, dim, lag, scale_w, scale_d, weight_type=False, use_cache=True, 
#         compress_size=250, distance_scale=1e0, n_filters=256, model_names=[], list_of_param_grids=[]
#         ):
#     data_bundle = dict()
#     if not use_cache:
#         data_bundle = compose(
#             load_data(dbname, dbpath),
#             remove_baseline(),
#             resample_ecg(fs_after=250),
#         )(data_bundle)
#     stats = defaultdict(list)

#     for random_state in range(42, 42+10):
#         rng = np.random.default_rng(random_state)
#         if use_cache:
#             data_bundle = load_cached_segments(data_bundle, dbname, random_state)
#         elif not use_cache:
#             data_bundle = compose(
#                 divide_segments(seg_dur=2, fs=250, minmax_scale=False, random_state=random_state, segs_per_person=1000),
#             )(data_bundle)
#         data_bundle = compose(
#             make_curves(dim=dim, lag=lag, reduce=0),  # time-delay embedding
#             compress_curves(size=compress_size),  # reduce
#         )(data_bundle)
#         clock = datetime.now()
#         if weight_type == 'w':
#             data_bundle = calculate_weights(scale=distance_scale)(data_bundle)
#         elif weight_type == 'm':
#             data_bundle = calculate_max_dispers(scale=distance_scale)(data_bundle)
#         stats[f'calc-time-{weight_type}'].append( (datetime.now()-clock).total_seconds() )
        
#         y = data_bundle['seg_ids']
#         idwise_cnt_normalized = np.sum([
#             (y == s)*np.cumsum((y == s) / np.sum(y == s))
#             for s in np.unique(y)
#         ], axis=0)
#         mask_train = (idwise_cnt_normalized < 0.8)
#         mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized >=0.6)
#         mask_test = (idwise_cnt_normalized > 0.8)
        
#         clock = datetime.now()
#         data_bundle = compose(
#             extract_distance(scale=scale_d, n_filters=n_filters, random_state=random_state),
#         )(data_bundle)
#         stats['calc-time-dist'].append( (datetime.now()-clock).total_seconds() )
#         X_dist = data_bundle['X_dist']
#         if not weight_type:
#             X = X_dist
#         else:
#             clock = datetime.now()
#             data_bundle = extract_fourier(scale=scale_w, n_filters=n_filters//2, weight_type=weight_type, random_state=random_state)(data_bundle)
#             stats[f'calc-time-fourier-{weight_type}'].append( (datetime.now()-clock).total_seconds() )
#             X_fourier = data_bundle[f'X_fourier_{weight_type}']
#             X = np.concatenate([X_dist[..., ::2], X_fourier], axis=1)

#         y_train, y_test = y[mask_train&~mask_val], y[mask_val]
#         X_train, X_test = X[mask_train&~mask_val], X[mask_val]

#         for model_name, param_grid in zip(model_names, list_of_param_grids):
#             write_log(model_name)
#             param_names = param_grid.keys()
#             for param_vals in itertools.product(*[param_grid[name] for name in param_names]):
#                 params = {name: val for name, val in zip(param_names, param_vals)}
#                 write_log(str(params))
#                 model = get_model(model_name, random_state=random_state, **params)
#                 model.fit(X_train, y_train)
#                 score = accuracy_score(y_test, model.predict(X_test))
#                 stats[(model_name,) + tuple(params.items())].append( score )
#                 write_log(score)
#             write_log('')
#         write_log('')

#     for model_name, param_grid in zip(model_names, list_of_param_grids):
#         write_log(f'result: {model_name}')
#         param_names = param_grid.keys()
#         params_max = None
#         score_max = 0
#         for param_vals in itertools.product(*[param_grid[name] for name in param_names]):
#             params = {name: val for name, val in zip(param_names, param_vals)}
#             score = np.mean(stats[(model_name,) + tuple(params.items())])
#             write_log(f'{model_name}, {params}, score={score}')
#             if score > score_max:
#                 score_max = score
#                 params_max = params
#         write_log(f'best if params={params_max} with score {score_max}')
# lag, scale_w, scale_d, weight_type=False, use_cache=True, 
#         compress_size=250, distance_scale=1e0, n_filters=256, model_names=[], list_of_param_grids=[]

        # dbname, dbpath, dim=3, lag=5, scale_ws=tuple(), scale_ds=tuple(), weight_type=False, 
        # downsample_size_small=100, downsample_size=200, distance_scale=1e0, n_filters_small=256, n_filters=1024,
        # model_names=[], param_grids=[],
        # split_no=42, use_cache=False, 


# def cache_features(dbname, random_state, dim, lag, compress_size, distance_scale, scale_w, scale_d, weight_type, n_filters):
#     data_bundle = dict()
#     data_bundle = load_cached_segments(data_bundle, dbname, random_state)
#     data_bundle = compose(
#         make_curves(dim=dim, lag=lag, reduce=0), 
#         compress_curves(size=compress_size), 
#         extract_distance(scale=scale_d, n_filters=n_filters, random_state=random_state)
#     )(data_bundle)
#     if weight_type == 'w':
#         data_bundle = compose(
#             calculate_weights(distance_scale),
#             extract_fourier(scale=scale_w, n_filters=n_filters//2, weight_type=weight_type, random_state=random_state),
#         )(data_bundle)
#     elif weight_type == 'm':
#         data_bundle = compose(
#             calculate_max_dispers(distance_scale),
#             extract_fourier(scale=scale_w, n_filters=n_filters//2, weight_type=weight_type, random_state=random_state),
#         )(data_bundle)
#     X = data_bundle['X_dist']
#     if weight_type:
#         X = np.concatenate([X[..., ::2], data_bundle[f'X_fourier_{weight_type}']], axis=-1)
    
#     file_name = '-'.join([
#         'seg_features', f'dbname={dbname}', f'random_state={random_state}', f'dim={dim}', f'lag={lag}',
#         f'compress_size={compress_size}', f'distance_scale={distance_scale}', f'scale_w={scale_w}', f'scale_d={scale_d}',
#         f'weight_type={weight_type}', f'n_filters={n_filters}'
#     ])
#     np.savez(file_name, X=X, y=data_bundle['seg_ids'])


def get_performances(
        dbname, dbpath, dim, lag, scale_w, scale_d, weight_type, 
        downsample_size=200, distance_scale=1e0, n_filters=1024, 
        model_names=[], list_of_params=[],
        split_no=42, use_cache=False, use_f_cache=False,
        train_ratio=0.8, without_dist=False, cross_examinations=None
    ):
    '''
    Main result of experiment is provided by this function.
    - `dbname`: one of 'FANTASIA' 'NSRDB' 'MITDB' 'AFDB'
    - `dbpath`: local path to the dataset downloaded from PhysioNet; unzipped folder is expected in the site referred by `dbpath`
    - `dim`: dimension of time-delay embedding
    - `lag`: lag of time-delay embedding, where the unit corresponds to one shift in array data
    - `scale_w`: radius of ball from which $\\xi$'s are sampled. $\\xi$ is used to compute Fourire coefficient
    - `scale_d': radius of ball from which 'landmarks' are sampled
    - `use_weighting`: one of False 'w' 'm'. If False, use landmarks only; if 'w', use landmarks and fourier coefficients of weighting;
      if 'm', use landmarks and fourier coefficients of diversifier
    - `use_cache`: always False; unsupported in released version
    '''
    stats = dict()
    data_bundle = dict()
    if use_cache:
        data_bundle = load_cached_segments(data_bundle, dbname, split_no)
    elif not use_cache:
        data_bundle = compose(
            load_data(dbname, dbpath),
            remove_baseline(),
            resample_ecg(fs_after=250),
            divide_segments(seg_dur=2, fs=250, minmax_scale=False, random_state=split_no, segs_per_person=1000),
        )(data_bundle)
    data_bundle = compose(
        make_curves(dim=dim, lag=lag, reduce=0),  # time-delay embedding
        compress_curves(size=downsample_size),  # reduce
    )(data_bundle)
    if weight_type == 'w':
        clock = datetime.now()
        data_bundle = calculate_weights(scale=distance_scale)(data_bundle)
        stats['calc-time-w'] = (datetime.now() - clock).total_seconds()
    elif weight_type == 'm':
        clock = datetime.now()
        data_bundle = calculate_max_dispers(scale=distance_scale)(data_bundle)
        stats['calc-time-m'] = (datetime.now() - clock).total_seconds()
    y = data_bundle['seg_ids']
    idwise_cnt_normalized = np.sum([
        (y == s)*np.cumsum((y == s) / np.sum(y == s))
        for s in np.unique(y)
    ], axis=0)
    mask_train = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized >= (0.8-train_ratio))
    # mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized >=0.6)
    mask_test = (idwise_cnt_normalized >= 0.8)

    random_state = split_no
    clock = datetime.now()
    data_bundle = compose(
        extract_distance(scale=scale_d, n_filters=n_filters, random_state=random_state),
    )(data_bundle)
    stats['calc-time-dist'] = (datetime.now() - clock).total_seconds()
    X_dist = data_bundle['X_dist']
    if weight_type:
        clock = datetime.now()
        if without_dist:
            data_bundle = extract_fourier(scale=scale_w, n_filters=n_filters, weight_type=weight_type, random_state=random_state)(data_bundle)
            stats[f'calc-time-fourier-{weight_type}'] = (datetime.now() - clock).total_seconds()
            X_fourier = data_bundle[f'X_fourier_{weight_type}']
            X = X_fourier
        else:
            data_bundle = extract_fourier(scale=scale_w, n_filters=n_filters//2, weight_type=weight_type, random_state=random_state)(data_bundle)
            stats[f'calc-time-fourier-{weight_type}'] = (datetime.now() - clock).total_seconds()
            X_fourier = data_bundle[f'X_fourier_{weight_type}']
            X = np.concatenate([X_dist[..., ::2], X_fourier], axis=1)
    else:
        X = X_dist
    y_train, y_test = y[mask_train], y[mask_test]
    X_train, X_test = X[mask_train], X[mask_test]

    warm_models = []
    for model_name, params in zip(model_names, list_of_params):
        write_log(model_name)
        model = get_model(model_name, random_state=random_state, **params)
        clock = datetime.now()
        model.fit(X_train, y_train)
        stats[f'calc-time-fit-{model_name}'] = (datetime.now() - clock).total_seconds()
        
        clock = datetime.now()
        score = accuracy_score(y_test, model.predict(X_test))
        stats[f'calc-time-pred-{model_name}'] = (datetime.now() - clock).total_seconds()
        stats[model_name] = score
        warm_models.append(model)
        write_log(score)
        
    write_log(f'performance: {stats}')
    write_log('')
    if not cross_examinations:
        return stats
    for ce in range(2):
        data_bundle = dict()
        if use_cache:
            data_bundle = load_cached_segments(data_bundle, dbname, split_no)
        elif not use_cache:
            data_bundle = compose(
                load_data(dbname, dbpath),
                remove_baseline(),
                resample_ecg(fs_after=250),
                divide_segments(seg_dur=2, fs=250, minmax_scale=False, random_state=split_no, segs_per_person=1000),
            )(data_bundle)
        if ce == 0: # make segment into 1.5s length
            data_bundle['segs'] = data_bundle['segs'][:, :375]
            data_bundle = compose(
                make_curves(dim=dim, lag=lag, reduce=0),  # time-delay embedding
                compress_curves(size=downsample_size),  # reduce
            )(data_bundle)
        elif ce == 1: # modify frequency to 500Hz
            from scipy.signal import resample
            data_bundle['segs'] = resample(data_bundle['segs'], 1000, axis=1)
            data_bundle = compose(
                make_curves(dim=dim, lag=lag*2, reduce=0),  # time-delay embedding
                compress_curves(size=downsample_size),  # reduce
            )(data_bundle)
        if weight_type == 'w':
            data_bundle = calculate_weights(scale=distance_scale)(data_bundle)
        elif weight_type == 'm':
            data_bundle = calculate_max_dispers(scale=distance_scale)(data_bundle)
        y = data_bundle['seg_ids']
        idwise_cnt_normalized = np.sum([
            (y == s)*np.cumsum((y == s) / np.sum(y == s))
            for s in np.unique(y)
        ], axis=0)
        mask_train = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized >= (0.8-train_ratio))
        # mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized >=0.6)
        mask_test = (idwise_cnt_normalized >= 0.8)

        random_state = split_no
        data_bundle = compose(
            extract_distance(scale=scale_d, n_filters=n_filters, random_state=random_state),
        )(data_bundle)
        X_dist = data_bundle['X_dist']
        if weight_type:
            if without_dist:
                data_bundle = extract_fourier(scale=scale_w, n_filters=n_filters, weight_type=weight_type, random_state=random_state)(data_bundle)
                X_fourier = data_bundle[f'X_fourier_{weight_type}']
                X = X_fourier
            else:
                data_bundle = extract_fourier(scale=scale_w, n_filters=n_filters//2, weight_type=weight_type, random_state=random_state)(data_bundle)
                X_fourier = data_bundle[f'X_fourier_{weight_type}']
                X = np.concatenate([X_dist[..., ::2], X_fourier], axis=1)
        else:
            X = X_dist
        y_train, y_test = y[mask_train], y[mask_test]
        X_train, X_test = X[mask_train], X[mask_test]

        for model_name, model in zip(model_names, warm_models):
            score = accuracy_score(y_test, model.predict(X_test))
            stats[f'{model_name}_C.E.{ce}'] = score
            write_log(f'C.E.{ce} result: {model_name}, score={score}')
    return stats




model_param_grids = {
    'knn': dict(
        n_neighbors=[1, 2, 5, 10],
        weights=['uniform', 'distance'],  # 2 - distance 
    ),
    'hgbt': dict(
        max_iter=[100, 200, 300], # 500, 1000?
        max_depth=[10, 12, 15], # 15, 20?
    ),
    'lr': dict(
        C=[1e-2, 1e-1, 1e0, 1e1, 1e2], # 3e-1 if not use, 3e0 if use
    ),
    'svm': dict(
        C=[1e-1, 1e0, 1e1],
        gamma=[1e-4, 1e-3, 1e-2],
    ),
    'mlp': dict(
        hidden_layer_sizes=[(64,), (128, ), (256, ), (512, ),],
    )
}

if __name__ == '__main__':
    dbnames = ['FANTASIA', 'NSRDB', 'MITDB', 'AFDB']
    dbpaths = ['D:/database', 'D:/database', 'D:/database', 'D:/database/mit-bih-atrial-fibrillation-database-1.0.0']

    # caching
    # for dbname, dbpath in zip(dbnames, dbpaths):
    #     cache_segments(dbname, dbpath, range(42, 42+10))
    
    # param selection and performance check
    for dbname, dbpath in zip(dbnames, dbpaths):
        exp_record = defaultdict(list)
        for split_no in range(42, 42+10):
            write_log(f'{dbname}-{split_no}')
            # parameter selection cross validation
            # scale_w, scale_d, list_of_best_params = select_geometric_params(
            #     dbname=dbname, dbpath=dbpath, 
            #     dim=3, lag=5, scale_ws=[0.25, 0.5, 1, 2, 4], scale_ds=[0.25, 0.5, 1, 2, 4], weight_type='w',
            #     downsample_size=100, distance_scale=1.0, n_filters_small=256, n_filters=1024,
            #     model_names=['lr', 'knn', 'svm', 'mlp'], 
            #     param_grids=[model_param_grids[key] for key in ['lr', 'knn', 'svm', 'mlp']],
            #     split_no=split_no, use_cache=True
            # )
            # used saved CV result
            scale_w, scale_d, list_of_best_params = load_validated_params('.validated.csv', dbname, split_no, 'w')
            # performance check: weighting case
            stats = get_performances(
                dbname=dbname, dbpath=dbpath, 
                dim=3, lag=5, scale_w=scale_w, scale_d=scale_d, weight_type='w', 
                downsample_size=200, distance_scale=1e0, n_filters=1024, 
                model_names=['lr', 'knn', 'svm', 'mlp'], 
                list_of_params=[list_of_best_params[key] for key in ['lr', 'knn', 'svm', 'mlp']],
                split_no=split_no, use_cache=True, use_f_cache=False,
                train_ratio=0.8
            )
            for key, val in stats.items():
                exp_record[f'score-{key}-weighting'].append(val)
                
            # performance check: diversifier case
            stats = get_performances(
                dbname=dbname, dbpath=dbpath, 
                dim=3, lag=5, scale_w=scale_w, scale_d=scale_d, weight_type='m', 
                downsample_size=200, distance_scale=1e0, n_filters=1024, 
                model_names=['lr', 'knn', 'svm', 'mlp'], 
                list_of_params=[list_of_best_params[key] for key in ['lr', 'knn', 'svm', 'mlp']],
                split_no=split_no, use_cache=True, use_f_cache=False,
                train_ratio=0.8
            )
            for key, val in stats.items():
                exp_record[f'score-{key}-diversifier'].append(val)

            # performance check: uniform measure case
            stats = get_performances(
                dbname=dbname, dbpath=dbpath, 
                dim=3, lag=5, scale_w=scale_w, scale_d=scale_d, weight_type='u', 
                downsample_size=200, distance_scale=1e0, n_filters=1024, 
                model_names=['lr', 'knn', 'svm', 'mlp'], 
                list_of_params=[list_of_best_params[key] for key in ['lr', 'knn', 'svm', 'mlp']],
                split_no=split_no, use_cache=True, use_f_cache=False,
                train_ratio=0.8
            )
            for key, val in stats.items():
                exp_record[f'score-{key}-uniform'].append(val)
                
            # parameter selection cross validation
            # scale_w, scale_d, list_of_best_params = select_geometric_params(
            #     dbname=dbname, dbpath=dbpath, 
            #     dim=3, lag=5, scale_ws=[0.1], scale_ds=[0.25, 0.5, 1, 2, 4], weight_type=None,
            #     downsample_size=100, distance_scale=1.0, n_filters_small=256, n_filters=1024,
            #     model_names=['lr', 'knn', 'svm', 'mlp'], 
            #     param_grids=[model_param_grids[key] for key in ['lr', 'knn', 'svm', 'mlp']],
            #     split_no=split_no, use_cache=True
            # )
            # used saved CV result
            scale_w, scale_d, list_of_best_params = load_validated_params('.validated.csv', dbname, split_no, None)
            # performance check: weighting case
            stats = get_performances(
                dbname=dbname, dbpath=dbpath, 
                dim=3, lag=5, scale_w=scale_w, scale_d=scale_d, weight_type=None, 
                downsample_size=200, distance_scale=1e0, n_filters=1024, 
                model_names=['lr', 'knn', 'svm', 'mlp'], 
                list_of_params=[list_of_best_params[key] for key in ['lr', 'knn', 'svm', 'mlp']],
                split_no=split_no, use_cache=True, use_f_cache=False,
                train_ratio=0.8
            )
            for key, val in stats.items():
                exp_record[f'score-{key}-default'].append(val)


        for key, val in exp_record.items():
            write_log(f'stats of {key}: mean={np.mean(val)}, std={np.std(val)}, max={np.max(val)}, min={np.min(val)}')
