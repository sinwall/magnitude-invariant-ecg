import itertools
from collections import defaultdict
from datetime import datetime
import logging
# import os
# import sys
# sys.path.append(os.path.abspath('../'))
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tools import (
    load_data, remove_baseline, resample_ecg, divide_segments,
    make_curves, compress_curves, calculate_weights, calculate_max_dispers,
    extract_fourier, extract_distance,
    split_into_train_test,
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

def get_model(name, random_state=None, **kwargs):
    if name == 'knn':
        kwargs_default = dict(
            n_neighbors=2,
            weights='distance',
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
            max_iter=100,
            max_depth=10,
            random_state=random_state
        )
        kwargs_default.update(kwargs)
        kwargs = kwargs_default
        model = HistGradientBoostingClassifier(
            **kwargs
        )
    elif name == 'lr':
        kwargs_default = dict(
            C=1e0,
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
    return model


def select_geometric_params():
    # lags = (5, )
    # scale_ws = (1e0, )
    # scale_ds = (1e0, )
    lags = (5, 10, 15)
    scale_ws = (2.5e-1, 5e-1, 1e0, 2e0, 4e0)
    scale_ds = (2.5e-1, 5e-1, 1e0, 2e0, 4e0)
    # prepare and preprocess data
    # and save into `data_bundle`
    for ds_name, ds_path in zip(
        ['MITDB', 'NSRDB', 'AFDB', 'FANTASIA'],
        ['E:/database', 'E:/database', 'E:/database/mit-bih-atrial-fibrillation-database-1.0.0', 'E:/database']
    ):
        write_log(f'dataset={ds_name}')
        data_bundle = dict()
        data_bundle = compose(
            load_data(ds_name, ds_path),
            remove_baseline(),
            resample_ecg(fs_after=250),
            divide_segments(seg_dur=2, fs=250, minmax_scale=False),
        )(data_bundle)

        # train/test/validation split
        y = data_bundle['seg_ids']
        idwise_cnt_normalized = np.sum([
            (y == s)*np.cumsum((y == s) / np.sum(y == s))
            for s in np.unique(y)
        ], axis=0)
        mask_train = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized > 0.4)
        mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized > 0.6)

        y_train, y_test = y[mask_train&~mask_val], y[mask_val]

        model = get_model('knn', n_neighbors=5)
        scores = defaultdict(list)
        for lag in lags:
            write_log(f'lag={lag}')
            data_bundle = compose(
                make_curves(dim=3, lag=lag, reduce=0),  # time-delay embedding
                compress_curves(size=400),  # reduce
                calculate_weights(scale=1e0),
            )(data_bundle)
            for scale_w, scale_d in itertools.product(scale_ws, scale_ds):
                write_log(f'scale_w={scale_w}, scale_d={scale_d}')
                for random_state in range(42, 42+5):
                    data_bundle = compose(
                        extract_fourier(scale=scale_w, n_filters=128, random_state=random_state),
                        extract_distance(scale=scale_d, n_filters=128, random_state=random_state),
                    )(data_bundle)
                    X_dist = data_bundle['X_dist']
                    X_fourier = data_bundle['X_fourier_w']
                    for use_weighting in [False, True]:
                        if not use_weighting:
                            X = X_dist
                        else:
                            X = np.concatenate([X_dist[..., ::2], X_fourier[..., ::2]], axis=1)
                        X_train, X_test = X[mask_train&~mask_val], X[mask_val]

                        model.fit(X_train, y_train)
                        score = accuracy_score(y_test, model.predict(X_test))
                        scores[(lag, scale_w, scale_d, use_weighting)].append( score )
                        write_log(score)
                    write_log('')
                write_log('')
            write_log('')
        for key in scores:
            scores[key] = np.mean(scores[key])
        for use_weighting in (False, True):
            write_log(f'If use_weighting={use_weighting}, validation result is:')
            results = [f'{key} {val}' for key, val in scores.items() if key[-1] == use_weighting]
            write_log('\n'.join(results))
            write_log(f'optimal with {results[np.argmax([el[1] for el in results])]}')
            write_log('')


def select_model_params(dbname, dbpath, dim=3, lag=5, scale_w=2e0, scale_d=1e0, use_weighting=False):
    data_bundle = dict()
    # prepare and preprocess data
    # and save into `data_bundle`
    data_bundle = compose(
        load_data(dbname, dbpath),
        remove_baseline(),
        resample_ecg(fs_after=250),
        divide_segments(seg_dur=2, fs=250),
        make_curves(dim=dim, lag=lag, reduce=0),  # time-delay embedding
        compress_curves(size=400),  # reduce
        calculate_weights(scale=1e0),
    )(data_bundle)


    model_param_grids = {
        'knn': dict(
            n_neighbors=[2, 5, 10, 20],
            weights=['uniform', 'distance'],  # 2 - distance 
        ),
        'hgbt': dict(
            max_iter=[100, 200, 300], # 500, 1000?
            max_depth=[10, 12, 15], # 15, 20?
        ),
        'lr': dict(
            C=[1e-1, 3e-1, 1e0, 3e0, 1e1], # 3e-1 if not use, 3e0 if use
        ),
    }

    # train/test/validation split
    y = data_bundle['seg_ids']
    idwise_cnt_normalized = np.sum([
        (y == s)*np.cumsum((y == s) / np.sum(y == s))
        for s in np.unique(y)
    ], axis=0)
    mask_train = (idwise_cnt_normalized < 0.8) # & (idwise_cnt_normalized > 0.7)
    mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized > 0.6)

    y_train, y_test = y[mask_train&~mask_val], y[mask_val]

    for model_name in ['lr']:
    # for model_name in ['knn', 'hgbt', 'lr']:
        write_log(model_name)
        scores = defaultdict(list)
        param_grid = model_param_grids[model_name]
        param_names = param_grid.keys()
        for param_vals in itertools.product(*[param_grid[name] for name in param_names]):
            params = {name: val for name, val in zip(param_names, param_vals)}
            for random_state in range(42, 42+5):
                model = get_model(model_name, random_state=random_state, **params)
                for use_weighting in [False, True]:
                    if not use_weighting:
                        data_bundle = compose(
                            extract_distance(scale=scale_d, n_filters=512, random_state=random_state),
                        )(data_bundle)
                        X_dist = data_bundle['X_dist']
                        X = X_dist
                    else:
                        data_bundle = compose(
                            extract_fourier(scale=scale_w, n_filters=512, random_state=random_state),
                            extract_distance(scale=scale_d, n_filters=512, random_state=random_state),
                        )(data_bundle)
                        X_dist = data_bundle['X_dist']
                        X_fourier = data_bundle['X_fourier_w']
                        X = np.concatenate([X_dist[..., ::2], X_fourier[..., ::2]], axis=1)
                    X_train, X_test = X[mask_train&~mask_val], X[mask_val]

                    model.fit(X_train, y_train)
                    score = accuracy_score(y_test, model.predict(X_test))
                    scores[(tuple(params.items()), use_weighting)].append( score )
                    write_log(score + ' ')
                write_log('')
            write_log('')
        write_log('')
        for key in scores:
            scores[key] = np.mean(scores[key])
        for use_weighting in (False, True):
            write_log(f'If use_weighting={use_weighting}, validation result is:')
            results = [f'{key} {val}' for key, val in scores.items() if key[-1] == use_weighting]
            write_log('\n'.join(results))
            write_log('')



def get_performances(dbname, dbpath, dim=3, lag=5, scale_w=2e0, scale_d=1e0, use_weighting=False):
    data_bundle = dict()
    # prepare and preprocess data
    # and save into `data_bundle`
    data_bundle = compose(
        load_data(dbname, dbpath),
        remove_baseline(),
        resample_ecg(fs_after=250),
        divide_segments(seg_dur=2, fs=250, minmax_scale=False),
        make_curves(dim=dim, lag=lag, reduce=0),  # time-delay embedding
        compress_curves(size=250),  # reduce
        # calculate_weights(scale=1e0),
        # calculate_max_dispers(scale=1e0),
    )(data_bundle)
    if use_weighting:
        data_bundle = calculate_weights(scale=1e0)(data_bundle)

    # train/test/validation split
    y = data_bundle['seg_ids']
    idwise_cnt_normalized = np.sum([
        (y == s)*np.cumsum((y == s) / np.sum(y == s))
        for s in np.unique(y)
    ], axis=0)
    mask_train = (idwise_cnt_normalized < 0.8) # & (idwise_cnt_normalized > 0.7)
    mask_test = idwise_cnt_normalized > 0.8
    # mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized > 0.7)

    y_train, y_test = y[mask_train], y[mask_test]
    scores = defaultdict(list)
    # for model_name in ['knn']:
    for model_name in ['knn', 'hgbt', 'lr']:
        write_log(model_name)
        for random_state in range(42, 42+10):
            if not use_weighting:
                data_bundle = compose(
                    extract_distance(scale=scale_d, n_filters=512, random_state=random_state)
                )(data_bundle)
                X_dist = data_bundle['X_dist']
                X = X_dist
            elif use_weighting == 'w':
                data_bundle = compose(
                    extract_fourier(scale=scale_w, n_filters=512, random_state=random_state),
                    extract_distance(scale=scale_d, n_filters=512, random_state=random_state),
                    # save_features(f'.cache_features-lag={lag}-scale_w={scale_w}-scale_d={scale_d}')
                )(data_bundle)
                X_dist = data_bundle['X_dist']
                X_fourier_w = data_bundle['X_fourier_w']
                X = np.concatenate([X_dist[...,::-2], X_fourier_w[..., ::-2]], axis=1)
            elif use_weighting == 'm':
                data_bundle = compose(
                    extract_fourier(scale=scale_w, n_filters=512, weight_type='m', random_state=random_state),
                    extract_distance(scale=scale_d, n_filters=512, random_state=random_state),
                )(data_bundle)
                X_dist = data_bundle['X_dist']
                X_fourier_m = data_bundle['X_fourier_m']
                X = np.concatenate([X_dist[...,::-2], X_fourier_m[..., ::-2]], axis=1)
            X_train, X_test = X[mask_train], X[mask_test]

            model = get_model(model_name, random_state=random_state)
            model.fit(X_train, y_train)
            score = accuracy_score(y_test, model.predict(X_test))
            scores[model_name].append( score )
            write_log(score)
        # print(f'result stats (wout w ): mean {np.mean(scores[(model_name, False)])}, std:{np.std(scores[(model_name, False)])}')
        if not use_weighting:
            write_log(f'result stats (wout w ): ')
        elif use_weighting == 'w':
            write_log(f'result stats (with w ): ')
        elif use_weighting == 'm':
            write_log(f'result stats (with w+): ')
        write_log(f'mean {np.mean(scores[model_name])}, std:{np.std(scores[model_name])}')
        write_log('')

# NSRDB
# optimal with ((10, 0.25, 1.0, False), 0.9585826965203623)
# optimal with ((5, 2.0, 0.5, True), 0.9592095084550929)

# FANTASIA
# optimal with ((5, 0.25, 4.0, False), 0.9949065238359088)
# optimal with ((5, 1.0, 1.0, True), 0.9964784830388476)

# MITDB
# optimal with ((5, 0.25, 2.0, False), 0.956809...)
# optimal with ((5, 1.0, 1.0, True), 0.961795...)

# AFDB
# optimal with ((5, 0.25, 2.0, False), 0.9638150232919255)
# optimal with ((5, 1.0, 0.5, True), 0.9680354716614905)


if __name__ == '__main__':
    # select_geometric_params()
    # select_model_params()
    get_performances('MITDB', 'E:/database', 5, 0.25, 2.0, False)
    get_performances('MITDB', 'E:/database', 5, 1.0, 1.0, 'w')
    get_performances('NSRDB', 'E:/database', 10, 0.25, 1.0, False)
    get_performances('NSRDB', 'E:/database', 5, 2.0, 0.5, 'w')
    get_performances('AFDB', 'E:/database/mit-bih-atrial-fibrillation-database-1.0.0', 5, 0.25, 2.0, False)
    get_performances('AFDB', 'E:/database/mit-bih-atrial-fibrillation-database-1.0.0', 5, 1.0, 0.5, 'w')
    get_performances('FANTASIA', 'E:/database', 5, 0.25, 4.0, False)
    get_performances('FANTASIA', 'E:/database', 5, 1.0, 1.0, 'w')
