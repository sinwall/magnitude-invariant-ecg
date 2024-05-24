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
        dbname, dbpath, dim=3, lag=5, scale_ws=tuple(), scale_ds=tuple(), weight_type=False, use_cache=True, 
        compress_size=250, distance_scale=1e0, n_filters=256, model_name=None
        ):
    '''
    This function is used to choose best `lag` and radii(`scale_w`, `scale_d`). 
    '''
    # prepare and preprocess data
    # and save into `data_bundle`
    data_bundle = dict()
    if not use_cache:
        data_bundle = compose(
            load_data(dbname, dbpath),
            remove_baseline(),
            resample_ecg(fs_after=250),
        )(data_bundle)
    model = get_model(model_name)
    stats = defaultdict(list)
    for random_state in range(42, 42+10):
        rng = np.random.default_rng(random_state)
        if use_cache:
            data_bundle = load_cached_segments(data_bundle, dbname, random_state)
        elif not use_cache:
            data_bundle = compose(
                divide_segments(seg_dur=2, fs=250, minmax_scale=False, random_state=random_state, segs_per_person=1000),
            )(data_bundle)
        data_bundle = compose(
            make_curves(dim=dim, lag=lag, reduce=0),  # time-delay embedding
            compress_curves(size=compress_size),  # reduce
        )(data_bundle)

        clock = datetime.now()
        if weight_type == 'w':
            data_bundle = calculate_weights(scale=distance_scale)(data_bundle)
        elif weight_type == 'm':
            data_bundle = calculate_max_dispers(scale=distance_scale)(data_bundle)
        stats[f'calc-time-{weight_type}'].append( (datetime.now()-clock).total_seconds() )

        y = data_bundle['seg_ids']
        idwise_cnt_normalized = np.sum([
            (y == s)*np.cumsum((y == s) / np.sum(y == s))
            for s in np.unique(y)
        ], axis=0)
        mask_train = (idwise_cnt_normalized < 0.8)
        mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized >=0.6)
        mask_test = (idwise_cnt_normalized > 0.8)

        for scale_d in scale_ds:
            clock = datetime.now()
            data_bundle = compose(
                extract_distance(scale=scale_d, n_filters=n_filters, random_state=random_state),
            )(data_bundle)
            stats['calc-time-dist'].append( (datetime.now()-clock).total_seconds() )
            X_dist = data_bundle['X_dist']

            for scale_w in scale_ws:
                clock = datetime.now()
                data_bundle = extract_fourier(scale=scale_w, n_filters=n_filters//2, weight_type=weight_type, random_state=random_state)(data_bundle)
                stats[f'calc-time-fourier-{weight_type}'].append( (datetime.now()-clock).total_seconds() )
                X_fourier = data_bundle[f'X_fourier_{weight_type}']
                if not weight_type:
                    X = X_dist
                else:
                    X = np.concatenate([X_dist[..., ::2], X_fourier], axis=1)
    
                y_train, y_test = y[mask_train&~mask_val], y[mask_test]
                X_train, X_test = X[mask_train&~mask_val], X[mask_test]

                clock = datetime.now()
                model.fit(X_train, y_train)
                stats['calc-time-fit'].append( (datetime.now()-clock).total_seconds() )

                clock = datetime.now()
                score = accuracy_score(y_test, model.predict(X_test))
                stats['calc-time-pred'].append( (datetime.now()-clock).total_seconds() )

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
    write_log(f'params_max={params_max}, score_max={score_max}')
    write_log('')


def select_model_params(
        dbname, dbpath, dim, lag, scale_w, scale_d, weight_type=False, use_cache=True, 
        compress_size=250, distance_scale=1e0, n_filters=256, model_names=[], list_of_param_grids=[]
        ):
    data_bundle = dict()
    if not use_cache:
        data_bundle = compose(
            load_data(dbname, dbpath),
            remove_baseline(),
            resample_ecg(fs_after=250),
        )(data_bundle)
    stats = defaultdict(list)

    for random_state in range(42, 42+10):
        rng = np.random.default_rng(random_state)
        if use_cache:
            data_bundle = load_cached_segments(data_bundle, dbname, random_state)
        elif not use_cache:
            data_bundle = compose(
                divide_segments(seg_dur=2, fs=250, minmax_scale=False, random_state=random_state, segs_per_person=1000),
            )(data_bundle)
        data_bundle = compose(
            make_curves(dim=dim, lag=lag, reduce=0),  # time-delay embedding
            compress_curves(size=compress_size),  # reduce
        )(data_bundle)
        clock = datetime.now()
        if weight_type == 'w':
            data_bundle = calculate_weights(scale=distance_scale)(data_bundle)
        elif weight_type == 'm':
            data_bundle = calculate_max_dispers(scale=distance_scale)(data_bundle)
        stats[f'calc-time-{weight_type}'].append( (datetime.now()-clock).total_seconds() )
        
        y = data_bundle['seg_ids']
        idwise_cnt_normalized = np.sum([
            (y == s)*np.cumsum((y == s) / np.sum(y == s))
            for s in np.unique(y)
        ], axis=0)
        mask_train = (idwise_cnt_normalized < 0.8)
        mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized >=0.6)
        mask_test = (idwise_cnt_normalized > 0.8)
        
        clock = datetime.now()
        data_bundle = compose(
            extract_distance(scale=scale_d, n_filters=n_filters, random_state=random_state),
        )(data_bundle)
        stats['calc-time-dist'].append( (datetime.now()-clock).total_seconds() )
        X_dist = data_bundle['X_dist']
        clock = datetime.now()
        data_bundle = extract_fourier(scale=scale_w, n_filters=n_filters//2, weight_type=weight_type, random_state=random_state)(data_bundle)
        stats[f'calc-time-fourier-{weight_type}'].append( (datetime.now()-clock).total_seconds() )
        X_fourier = data_bundle[f'X_fourier_{weight_type}']
        if not weight_type:
            X = X_dist
        else:
            X = np.concatenate([X_dist[..., ::2], X_fourier], axis=1)

        y_train, y_test = y[mask_train&~mask_val], y[mask_test]
        X_train, X_test = X[mask_train&~mask_val], X[mask_test]

        for model_name, param_grid in zip(model_names, list_of_param_grids):
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

    for model_name, param_grid in zip(model_names, list_of_param_grids):
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
        write_log(f'best if params={params_max} with score {score_max}')

def get_performances(
        dbname, dbpath, dim=3, lag=5, scale_w=2e0, scale_d=1e0, use_weighting=False, use_cache=False, train_ratio=0.8,
        compress_size=250, distance_scale=1e0
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
    data_bundle = dict()
    # prepare and preprocess data
    # and save into `data_bundle`
    if not use_cache:
        data_bundle = compose(
            load_data(dbname, dbpath),
            remove_baseline(),
            resample_ecg(fs_after=250),
            divide_segments(seg_dur=2, fs=250, minmax_scale=False),
            make_curves(dim=dim, lag=lag, reduce=0),  # time-delay embedding
            compress_curves(size=compress_size),  # reduce
            # calculate_weights(scale=1e0),
            # calculate_max_dispers(scale=1e0),
        )(data_bundle)
        if use_weighting == 'w':
            data_bundle = calculate_weights(scale=distance_scale)(data_bundle)
        elif use_weighting == 'm':
            data_bundle = calculate_max_dispers(scale=distance_scale)(data_bundle)

        # train/test/validation split
        y = data_bundle['seg_ids']
        idwise_cnt_normalized = np.sum([
            (y == s)*np.cumsum((y == s) / np.sum(y == s))
            for s in np.unique(y)
        ], axis=0)
        mask_train = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized >= (0.8-train_ratio))
        mask_test = idwise_cnt_normalized > 0.8
        # mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized > 0.7)

        y_train, y_test = y[mask_train], y[mask_test]
    scores = defaultdict(list)
    for model_name in ['lr']:
    # for model_name in ['knn', 'hgbt']:
        write_log(model_name)
        for random_state in range(42, 42+10):
            if not use_weighting:
                data_bundle = compose(
                    extract_distance(scale=scale_d, n_filters=512, random_state=random_state)
                )(data_bundle)
                X_dist = data_bundle['X_dist']
                X = X_dist
            elif use_cache:
                npz_loaded = np.load(f'.cache_features-dbname={dbname}-lag={lag}-scale_w={scale_w}-scale_d={scale_d}-random_state={random_state}.npz')
                X_dist = npz_loaded['X_dist']
                X_fourier_w = npz_loaded['X_fourier_w']
                # train/test/validation split
                y = npz_loaded['y']
                idwise_cnt_normalized = np.sum([
                    (y == s)*np.cumsum((y == s) / np.sum(y == s))
                    for s in np.unique(y)
                ], axis=0)
                mask_train = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized >= (0.8-train_ratio))
                mask_test = idwise_cnt_normalized > 0.8
                # mask_val = (idwise_cnt_normalized < 0.8) & (idwise_cnt_normalized > 0.7)

                y_train, y_test = y[mask_train], y[mask_test]
                X = np.concatenate([X_dist[...,::-2], X_fourier_w[..., ::-2]], axis=1)
            elif use_weighting == 'w':
                data_bundle = compose(
                    extract_fourier(scale=scale_w, n_filters=512, random_state=random_state),
                    extract_distance(scale=scale_d, n_filters=512, random_state=random_state),
                    save_features(f'.cache_features-dbname={dbname}-lag={lag}-scale_w={scale_w}-scale_d={scale_d}-random_state={random_state}.npz')
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
        C=[1e-1, 3e-1, 1e0, 3e0, 1e1], # 3e-1 if not use, 3e0 if use
    ),
    'svm': dict(
        C=[1e-1, 1e0, 1e1],
        gamma=[1e-3, 1e-2, 1e-1],
    ),
    'mlp': dict(
        hidden_layer_sizes=[(64,), (128, ), (256, ), (512, ),],
    )
}

if __name__ == '__main__':
    # caching
    # for dbname, dbpath in zip(
    #     ['MITDB', 'NSRDB', 'AFDB', 'FANTASIA'],
    #     ['E:/database', 'E:/database', 'E:/database/mit-bih-atrial-fibrillation-database-1.0.0', 'E:/database']
    # ):
    #     cache_segments(dbname, dbpath, range(42, 42+10))
    
    # geometric-params
    # for dbname, dbpath in zip(
    #     ['MITDB', 'NSRDB', 'AFDB', 'FANTASIA'],
    #     ['E:/database', 'E:/database', 'E:/database/mit-bih-atrial-fibrillation-database-1.0.0', 'E:/database']
    # ):
    #     write_log(f'select_geometric_params {dbname}')
    #     select_geometric_params(
    #         dbname, dbpath,
    #         3, 5, (0.25, 0.5, 1, 2, 4), (0.25, 0.5, 1, 2, 4), 'w',
    #         use_cache=True, compress_size=100, distance_scale=1, n_filters=256, model_name='knn'
    #     )

    # model-params
    for dbname, dbpath, scale_w, scale_d in zip(
        ['MITDB', 'NSRDB', 'AFDB', 'FANTASIA'],
        ['E:/database', 'E:/database', 'E:/database/mit-bih-atrial-fibrillation-database-1.0.0', 'E:/database'],
        [1., 2., 2., 1.,],
        [1., .5, .5, 1.]
    ):
        write_log(f'model_params {dbname}')
        select_model_params(
            dbname, dbpath, 3, 5, scale_w, scale_d, 'w',
            compress_size=100, distance_scale=1, n_filters=1024, 
            model_names=['lr', 'knn', 'svm', 'mlp'],
            list_of_param_grids=[model_param_grids[key] for key in ['lr', 'knn', 'svm', 'mlp']]
        )


    # for dbname, dbpath, scale_w, scale_d in zip(
    #     ['MITDB', 'NSRDB', 'AFDB', 'FANTASIA'],
    #     ['E:/database', 'E:/database', 'E:/database/mit-bih-atrial-fibrillation-database-1.0.0', 'E:/database'],
    #     [1., 2., 1., 1.,],
    #     [1., .5, .5, 1.]
    # ):
    #     write_log(f'experiment with {dbname}, no weighting,, now with comp=250, hgbt&lr')
    #     select_geometric_params(dbname, dbpath, 3, 5, 1, 1, False, False, 250, model_name='hgbt')
    #     select_geometric_params(dbname, dbpath, 3, 5, 1, 1, False, False, 250, model_name='lr')

    # select_geometric_params()
    # select_model_params()
    # get_performances('MITDB', 'E:/database', 5, 0.25, 2.0, False)
    # get_performances('MITDB', 'E:/database', 5, 1.0, 1.0, 'w', False)
    # get_performances('NSRDB', 'E:/database', 10, 0.25, 1.0, False)
    # get_performances('NSRDB', 'E:/database', 3, 5, 2.0, 0.5, 'w')
    # write_log('logistic regression - FANTASIA')
    # get_performances('FANTASIA', 'E:/database', 3, 5, 0.25, 4.0, False)
    # get_performances('FANTASIA', 'E:/database', 3, 5, 1.0, 1.0, 'w', True)
    # write_log('logistic regression - AFDB')
    # get_performances('AFDB', 'E:/database/mit-bih-atrial-fibrillation-database-1.0.0', 3, 5, 0.25, 2.0, False, False)
    # get_performances('AFDB', 'E:/database/mit-bih-atrial-fibrillation-database-1.0.0', 3, 5, 1.0, 0.5, 'w', True)

    # write_log('MITDB for cache generation')
    # get_performances('MITDB', 'E:/database', 3, 5, 1.0, 1.0, 'w', False)
    # for train_ratio in [0.6, 0.4, 0.2, 0.7, 0.5, 0.3]:
    #     write_log(f'train_ratio={train_ratio}')
    #     get_performances('MITDB', 'E:/database', 3, 5, 1.0, 1.0, 'w', True, train_ratio)
    #     get_performances('FANTASIA', 'E:/database', 3, 5, 1.0, 1.0, 'w', True, train_ratio)