import os
import wfdb
import numpy as np
from scipy.spatial import distance_matrix
from scipy.signal import butter, resample, sosfilt
from numba import jit, prange


class Embedder():
    def __init__(self, dim=2, lag=1, reduce=1, dim_raw=None, channel_last=False):
        self.dim = dim
        self.lag = lag
        self.reduce = reduce
        if dim_raw is None:
            dim_raw = dim + reduce
        self.dim_raw = dim_raw
        self.channel_last = channel_last
        A = np.stack([np.ones((dim_raw,))] + [np.linspace(0, 1, dim_raw) ** i for i in range(1, reduce)], axis=1)
        self.proj = np.linalg.svd(A)[0][:, reduce:]

    def transform(self, X):
        dim, lag, reduce, dim_raw = self.dim, self.lag, self.reduce, self.dim_raw
        if self.channel_last:
            length = X.shape[-2]
            result = np.stack([X[..., i*lag:length-(dim_raw-i-1)*lag, :] for i in range(dim_raw)], axis=-1)
        else:
            length = X.shape[-1]
            result = np.stack([X[..., i*lag:length-(dim_raw-i-1)*lag] for i in range(dim_raw)], axis=-1)
        if reduce > 0:
            result = result @ self.proj
        if self.channel_last:
            result = result.reshape(result.shape[:-2] + (-1, ))
        return result

    @property
    def length_loss(self):
        return (self.dim-1)*self.lag


class Weighting():
    def __init__(self, unit=1, method=None):
        if method is None:
            method = 'identity'
        self.unit = unit
        self.method = method

    def apply(self, pts):
        n_pts = pts.shape[0]
        dist_mat = distance_matrix(pts, pts)
        if self.method == 'identity':
            A = dist_mat
        elif self.method == 'exp':
            A = np.exp(-dist_mat / self.unit)
        b = np.ones(n_pts)
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        if self.method == 'identity':
            w /= (np.sum(w))**2
        return w.reshape(-1, 1)


@jit(nopython=True, parallel=True)
def _calculate_weighting_vectors(pts_ary):
    list_size = pts_ary.shape[0]
    result = np.empty(pts_ary.shape[:-1])
    n_pts = result.shape[-1]
    dim = pts_ary.shape[-1]
    for i in prange(list_size):
        pts = pts_ary[i]
        # pts_diff = np.empty((n_pts, n_pts, dim))
        # for j in prange(n_pts):
        #     pts_diff[j] = pts
        # for k in prange(n_pts):
        #     pts_diff[:, k] -= pts
        pts_diff = np.zeros((n_pts, n_pts, dim))
        for j in prange(n_pts):
            pts_diff[j] += pts
            pts_diff[:, j] -= pts
        dist_mat = np.sqrt(np.sum(np.square(pts_diff), axis=-1))
        A = np.exp(-dist_mat)
        b = np.ones(n_pts)
        w = np.linalg.lstsq(A, b, rcond=-1)[0]
        result[i] = w
    return result


def calculate_weighting_vectors(pts_lst, scale=1.):
    if isinstance(pts_lst, np.ndarray):
        shape = pts_lst.shape
        pts_lst = pts_lst.reshape((-1, ) + shape[-2:])
        result = _calculate_weighting_vectors(pts_lst/scale).reshape(shape[:-1])
    elif isinstance(pts_lst, list):
        result = [_calculate_weighting_vectors(pts[np.newaxis]/scale)[0] for pts in pts_lst]
    else:
        raise ValueError
    return result


@jit(nopython=True, parallel=True)
def _calculate_maximum_dipersion(pts_ary):
    list_size = pts_ary.shape[0]
    result = np.empty(pts_ary.shape[:-1])
    n_pts = result.shape[-1]
    dim = pts_ary.shape[-1]
    for i in prange(list_size):
        pts = pts_ary[i]
        # pts_diff = np.empty((n_pts, n_pts, dim))
        # for j in prange(n_pts):
        #     pts_diff[j] = pts
        # for k in prange(n_pts):
        #     pts_diff[:, k] -= pts
        pts_diff = np.zeros((n_pts, n_pts, dim))
        for j in prange(n_pts):
            pts_diff[j] += pts
            pts_diff[:, j] -= pts
        dist_mat = np.sqrt(np.sum(np.square(pts_diff), axis=-1))
        A = np.exp(-dist_mat)
        b = np.ones(n_pts)
        w = np.linalg.lstsq(A, b, rcond=-1)[0]
        while np.any(w < 0):
            mask_nonneg_w = (w > 0)
            w[mask_nonneg_w] = np.linalg.lstsq(
                A[mask_nonneg_w][:, mask_nonneg_w], b[mask_nonneg_w], rcond=-1)[0]
            w[~mask_nonneg_w] = 0
        result[i] = w
    return result

import cvxpy as cp
def calculate_maximum_dispersion(pts_lst, scale=1.):
    result = []
    for pts in pts_lst:
        dist_mat = np.linalg.norm(pts[..., np.newaxis, :] - pts, axis=-1)
        zeta_mat = np.exp(-dist_mat / scale)
        ones_vec = np.ones(zeta_mat.shape[:1])

        w = cp.Variable(zeta_mat.shape[0])
        prob = cp.Problem(
            cp.Minimize(w@zeta_mat@w),
            [w@ones_vec == 1, (-w) <= 0]
        )
        prob.solve()
        result.append(w.value)
    if isinstance(pts_lst, np.ndarray):
        result = np.array(result)
    return result

# def calculate_maximum_dispersion(pts_lst, scale=1.):
#     if isinstance(pts_lst, np.ndarray):
#         shape = pts_lst.shape
#         pts_lst = pts_lst.reshape((-1, ) + shape[-2:])
#         result = _calculate_maximum_dipersion(pts_lst/scale).reshape(shape[:-1])
#     elif isinstance(pts_lst, list):
#         result = [_calculate_maximum_dipersion(pts[np.newaxis]/scale) for pts in pts_lst]
#     else:
#         raise ValueError
#     return result


class SineFilter():
    def __init__(self, dim, n_filters, scale=1., random_state=None):
        rng = np.random.default_rng(random_state)
        self._wave_numbers = (2*rng.random((dim, n_filters))-1)
        self._wave_numbers *= (1/scale)*rng.random((1, n_filters)) / np.linalg.norm(self._wave_numbers, axis=0, keepdims=True)
        self.random_state = random_state

    def apply(self, pts, weights, batch_size=None):
        if batch_size is None:
            batch_size = pts.shape[0]
        if pts.ndim > weights.ndim:
            weights = weights[..., np.newaxis]
        result = np.empty((pts.shape[0], self._wave_numbers.shape[1]))
        for i_start in range(0, result.shape[0], batch_size):
            i_end = i_start + batch_size
            pts_batch = pts[i_start:i_end]
            weights_batch = weights[i_start:i_end]
            result[i_start:i_end] = np.sum(np.sin(pts_batch @ self._wave_numbers)*weights_batch, axis=-2)
        return result


class FourierFilter():
    def __init__(self, dim, n_filters, scale=1., random_state=None):
        self.dim = dim
        self.n_filters = n_filters
        self.scale = scale
        self.random_state = random_state
        rng = np.random.default_rng(random_state)
        n_wave_nums = -(-n_filters // 2)
        # n_wave_nums = n_filters
        # self.wave_numbers = (2*rng.random((n_wave_nums, dim))-1)
        # self.wave_numbers *= rng.random((n_wave_nums, 1)) / np.linalg.norm(self.wave_numbers, axis=1, keepdims=True)
        self.wave_numbers = (2*np.pi/scale)*uniform_sampling_sphere(rng, dim, n_wave_nums)

    def apply(self, pts, weights, batch_size=None):
        if batch_size is None:
            batch_size = pts.shape[0]
        # if pts.ndim > weights.ndim:
        #     weights = weights[..., np.newaxis]
        weights = np.squeeze(weights)
        weights = weights.astype(complex)
        return _fourier_filter_apply(pts, weights, self.n_filters, self.wave_numbers)


def uniform_sampling_sphere(rng, dim, n_pts):
    result = np.empty((0, dim))
    while result.shape[0] < n_pts:
        append = -1+2*rng.random((n_pts, dim))
        append = append[np.linalg.norm(append, axis=1) <= 1]
        result = np.concatenate([result, append])
    result = result[:n_pts]
    return result

@jit(nopython=True, parallel=True)
def _fourier_filter_apply(pts, weights, n_filters, wave_numbers):
    n_wave_nums = wave_numbers.shape[0]
    result = np.empty((pts.shape[0], n_filters))
    for i in prange(result.shape[0]):
        complex_fourier = np.exp(1j*(pts[i] @ wave_numbers.T)).T @ weights[i]
        result[i, :n_wave_nums] = np.real(complex_fourier)
        result[i, n_wave_nums:] = np.imag(complex_fourier[:n_filters-n_wave_nums])
    return result


class DistFilter():
    def __init__(self, dim, n_filters, scale=1., random_state=None):
        self.dim = dim
        self.n_filters = n_filters
        self.scale = scale
        self.random_state = random_state
        rng = np.random.default_rng(random_state)
        self.landmarks = (rng.random((n_filters, dim))*2-1)*scale
        self.min_or_max = rng.integers(0, 2, (n_filters, ))*2-1
        # self.min_or_max = np.ones_like(self.min_or_max)

    
    def apply(self, pts, weights=None, batch_size=None):
        return _apply_naive_filter(pts, self.landmarks, self.min_or_max)

@jit(nopython=True, parallel=True)
def _apply_naive_filter(curves, landmarks, min_or_max):
    result = np.empty((curves.shape[0], landmarks.shape[0]))
    for i in prange(curves.shape[0]):
        for j in prange(landmarks.shape[0]):
            result[i, j] = np.min(min_or_max[j]*np.sum(np.square(curves[i] - landmarks[j]), axis=1))
    for j in prange(landmarks.shape[0]):
        result[:, j] *= min_or_max[j]
    return np.sqrt(result)


symb_to_AAMI = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V', 
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}


def as_partial(func):
    def _decorated(*args, **kwargs):
        def _inner(x):
            return func(x, *args, **kwargs)
        return _inner
    return _decorated


@as_partial
def load_data(data_bundle, dataset_name, dataset_path):
    if data_bundle is None:
        data_bundle = dict()
    if dataset_name == 'MITDB':
        ecg_signals, ecg_ids = _load_MIT_BIH(input_path=dataset_path)
        fs = 360
    elif dataset_name == 'ECG-ID':
        ecg_signals, ecg_ids = _load_ECG_ID(input_path=dataset_path)
        fs = 500
    elif dataset_name == 'PTB':
        ecg_signals, ecg_ids = _load_PTB(input_path=dataset_path)
        fs = 1000
    elif dataset_name == 'NSRDB':
        ecg_signals, ecg_ids = _load_NSRDB(input_path=dataset_path)
        fs = 128
    elif dataset_name == 'AFDB':
        ecg_signals, ecg_ids = _load_AFDB(input_path=dataset_path)
        fs = 250
    elif dataset_name == 'FANTASIA':
        ecg_signals, ecg_ids = _load_FANTAISIA(input_path=dataset_path)
        fs = 250
    else:
        raise ValueError('Dataset name not understood.')

    data_bundle['ecg_raw'] = ecg_signals
    data_bundle['ecg_signals'] = ecg_signals
    data_bundle['ecg_ids'] = ecg_ids
    data_bundle['fs'] = fs
    return data_bundle

def _load_MIT_BIH(input_path):
    input_path = os.path.join(input_path, 'mit-bih-arrhythmia-database-1.0.0')
    ecg_signals = []
    ecg_ids = []
    ann_symbols = []
    ann_locs = []

    for file_name in sorted(os.listdir(f'{input_path}')):
        if not file_name.endswith('hea'):
            continue
        file_name = f'{input_path}/{file_name[:-4]}'
        sig, info = wfdb.rdsamp(file_name)
        atr = wfdb.rdann(file_name, 'atr')
        # ecg_signals.append( sig ); ecg_ids.append( int(file_name[-3:]) )
        ecg_signals.append( sig[:, 0] ); ecg_ids.append( int(file_name[-3:]) )
        ann_locs.append( np.array(atr.sample) )
        ann_symbols.append( np.array([symb_to_AAMI.get(el, el) for el in atr.symbol]) )
    ecg_signals = np.stack(ecg_signals); ecg_ids = np.array(ecg_ids)
    return ecg_signals, ecg_ids

def _load_ECG_ID(input_path):
    input_path = os.path.join(input_path, 'ecg-id-database-1.0.0')
    ecg_signals = []
    ecg_ids = []
    for person_num in range(1, 90+1):
        header_names = sorted(
            filter(
                lambda name: name.endswith('.hea'),
                os.listdir(f'{input_path}/Person_{person_num:0>2}')
            ),
            key=lambda name: int(name.split('_')[1].split('.')[0])
        )
        for header_name in header_names:
            sig, info = wfdb.rdsamp(f'{input_path}/Person_{person_num:0>2}/{header_name[:-4]}')
            atr = wfdb.rdann(f'{input_path}/Person_{person_num:0>2}/{header_name[:-4]}', 'atr')
            ecg_signals.append( sig[:, 1])
            ecg_ids.append(person_num)
    ecg_signals = np.stack(ecg_signals); ecg_ids = np.array(ecg_ids)
    return ecg_signals, ecg_ids

def _load_PTB(input_path):
    input_path = os.path.join(input_path, 'ptb-diagnostic-ecg-database-1.0.0')
    ecg_signals = []
    ecg_ids = []
    for patient_num in range(1, 294+1):
        if not os.path.isdir(f'{input_path}/patient{patient_num:0>3}'): continue
        header_names = sorted(
            filter(
                lambda name: name.endswith('.hea'),
                os.listdir(f'{input_path}/patient{patient_num:0>3}')
            ),
        )
        for header_name in header_names:
            sig, info = wfdb.rdsamp(f'{input_path}/patient{patient_num:0>3}/{header_name[:-4]}')
            # atr = wfdb.rdann(f'{input_path}/patient{patient_num:0>3}/{header_name[:-4]}', 'atr')
            ecg_signals.append( sig[:, 1])
            ecg_ids.append(patient_num)
    ecg_signals = np.stack(ecg_signals); ecg_ids = np.array(ecg_ids)
    return ecg_signals, ecg_ids

def _load_NSRDB(input_path):
    input_path = os.path.join(input_path, 'mit-bih-normal-sinus-rhythm-database-1.0.0')
    ecg_signals = []
    ecg_ids = []
    header_names = sorted(
        filter(
            lambda name: name.endswith('.hea'),
            os.listdir(f'{input_path}')
        ),
    )
    for header_name in header_names:
        sig, info = wfdb.rdsamp(f'{input_path}/{header_name[:-4]}')
        atr = wfdb.rdann(f'{input_path}/{header_name[:-4]}', 'atr')
        for idx_first_N, symb_first in zip(atr.sample, atr.symbol):
            if symb_first == 'N': break
        for idx_last_N, symb_last in zip(reversed(atr.sample,), reversed(atr.symbol)):
            if symb_last == 'N': break
        sig = sig[idx_first_N:idx_last_N]
        ecg_signals.append( sig[:, 0])
        ecg_ids.append(int(header_name[:-4]))
    ecg_ids = np.array(ecg_ids)
    return ecg_signals, ecg_ids


def _load_FANTAISIA(input_path):
    data_path = os.path.join(input_path, 'fantasia-database-1.0.0')
    ecg_signals = []
    ecg_ids = []
    for file_name in sorted(os.listdir(data_path)):
        if not file_name.endswith('.dat'): continue
        if 'f2o02' in file_name: continue
        sig, info = wfdb.rdsamp(os.path.join(data_path, file_name[:-4]))
        sig = sig[:, 1]
        # sig = sig[~np.isnan(sig)]
        if info['fs'] != 250:
            sig = resample(
                sig, 
                num=int(len(sig)*(250/info['fs'])),
                axis=0
            )
        ecg_signals.append( sig )
        ecg_ids.append(file_name[:-4])
    ecg_ids = np.array(ecg_ids)
    return ecg_signals, ecg_ids


def _load_AFDB(input_path):
    data_path = os.path.join(input_path, 'files')
    ecg_signals = []
    ecg_ids = []
    for file_name in sorted(os.listdir(data_path)):
        if not file_name.endswith('dat'): continue
        file_name_full = os.path.join(data_path, file_name[:-4])
        sig, info = wfdb.rdsamp(file_name_full)
        atr = wfdb.rdann(file_name_full, 'qrs')
        assert info['fs'] == 250
        sig = sig[:atr.sample[-1], 0]
        ecg_signals.append(sig)
        ecg_ids.append(int(file_name[:-4]))
    ecg_ids = np.array(ecg_ids)
    return ecg_signals, ecg_ids


@as_partial
def remove_baseline(data_bundle, sos=None):
    ecg_signals = data_bundle['ecg_signals']
    fs = data_bundle['fs']
    if sos is None:
        sos = butter(4, [0.5, 50], btype='bandpass', output='sos', fs=fs)
    
    if isinstance(ecg_signals, np.ndarray):
        ecg_filtered = sosfilt(
            sos,
            ecg_signals,
            axis=1
        )
    else:
        ecg_filtered = [sosfilt(
            sos,
            ecg_signal,
            axis=0
        ) for ecg_signal in ecg_signals]
    data_bundle['ecg_signals'] = ecg_filtered
    return data_bundle


@as_partial
def resample_ecg(data_bundle, fs_after=250):
    ecg_signals = data_bundle['ecg_signals']
    fs_before = data_bundle['fs']
    if fs_after == fs_before:
        return data_bundle
    if isinstance(ecg_signals, np.ndarray):
        ecg_resampled = resample(
            ecg_signals, 
            num=int(ecg_signals.shape[1]*(fs_after/fs_before)),
            axis=1
        )
    else:
        ecg_resampled = [
            resample(
                ecg_signal, 
                num=int(ecg_signal.shape[0]*(fs_after/fs_before)),
                axis=0
            ) for ecg_signal in ecg_signals
        ]
    data_bundle['ecg_signals'] = ecg_resampled
    return data_bundle

@as_partial
def divide_segments(data_bundle, seg_dur=2, fs=250, ol_rate=0, minmax_scale=False):
    ecg_signals = data_bundle['ecg_signals']
    ecg_ids = data_bundle['ecg_ids']
    seg_len = int(seg_dur*fs)
    segs = []; seg_ids = []
    if isinstance(ecg_signals, np.ndarray):
        raw_len = ecg_signals.shape[1]
        for i in range(0, raw_len, int((1-ol_rate)*fs)):
            seg = ecg_signals[:, i:i+seg_len]
            if seg.shape[1] < seg_len: break
            segs.append( seg )
            seg_ids.append( ecg_ids )
        segs = np.swapaxes(segs, 1, 0).reshape((-1, seg_len) + ecg_signals.shape[2:])
        seg_ids = np.swapaxes(seg_ids, 1, 0).reshape(-1)
    else:
        for ecg_signal, ecg_id in zip(ecg_signals, ecg_ids):
            raw_len = ecg_signal.shape[0]
            for i in range(0, raw_len, int((1-ol_rate)*fs)):
                seg = ecg_signal[i:i+seg_len]
                if seg.shape[0] < seg_len: break
                segs.append( seg )
                seg_ids.append( ecg_id )
        segs = np.array(segs); seg_ids = np.array(seg_ids)
    mask_na = np.any(np.isnan(segs), axis=1)
    segs = segs[~mask_na]
    seg_ids = seg_ids[~mask_na]
    if minmax_scale:
        segs_min = np.min(segs, axis=1, keepdims=True)
        segs_max = np.max(segs, axis=1, keepdims=True)
        segs = -1 + 2*(segs - segs_min) / (segs_max - segs_min)
    data_bundle['segs'] = segs
    data_bundle['seg_ids'] = seg_ids
    return data_bundle

@as_partial
def make_curves(data_bundle, dim=3, lag=4, reduce=0):
    segs = data_bundle['segs']
    embedder = Embedder(dim=dim, lag=lag, reduce=reduce)
    curves = embedder.transform(segs)
    data_bundle['curves'] = curves
    return data_bundle

@as_partial
def compress_curves(data_bundle, size):
    curves = data_bundle['curves']
    data_bundle['curves_uncompressed'] = curves
    size_r = size
    batch_size = 1000000
    print('Compression countdown started.', end=' ')
    while size_r > 0:
        curves_comp = []
        print(size_r, end=' ')
        n_remove = size_r // 2 if size_r > 1 else 1
        size_r -= n_remove
        for i in range(0, curves.shape[0], batch_size):
            curves_batch = curves[i:i+batch_size]
            curves_comp.append( _compress_onestep(curves_batch, n_remove) )
        curves = np.concatenate(curves_comp, axis=0)
    print()
    data_bundle['curves'] = curves
    return data_bundle

@jit(nopython=True, parallel=True)
def _compress_onestep(curves, size):
    n_timesteps = curves.shape[-2]
    n_remove = size
    diff_norm = np.sum(np.square(curves[:, 0:n_timesteps-n_timesteps%2:2] - curves[:, 1::2]), axis=-1)
    # diff_norm = np.linalg.norm(curves[:, 0:n_timesteps-n_timesteps%2:2] - curves[:, 1::2], axis=-1)
    # keepidxs = np.empty((curves.shape[0], n_timesteps-n_remove), dtype=np.int32)
    # keepidxs[:, n_timesteps//2 - n_remove:n_timesteps-n_timesteps%2 - n_remove] = np.arange(1, curves.shape[1], 2)
    # if (n_timesteps % 2 != 0):
    #     keepidxs[:, -1] = n_timesteps-1

    result = np.empty((curves.shape[0], n_timesteps-n_remove, curves.shape[-1]))
    for i in prange(curves.shape[0]):
        keepidxs = np.empty((n_timesteps-n_remove, ), dtype=np.int32)
        keepidxs[n_timesteps//2 - n_remove:n_timesteps-n_timesteps%2 - n_remove] = np.arange(1, curves.shape[1], 2)
        if (n_timesteps % 2 != 0):
            keepidxs[-1] = n_timesteps-1
        keepidxs[:n_timesteps//2 - n_remove] = 2*np.argsort(diff_norm[i])[n_remove:]
        keepidxs = np.sort(keepidxs)
        result[i] = curves[i, keepidxs]
    return result
    # keepidxs[:, :n_timesteps//2 - n_remove] = 2*np.argpartition(diff_norm, n_remove, axis=1)[:, n_remove:]
    # keepidxs = np.sort(keepidxs, axis=1)
    # curves = np.stack([
    #     np.take_along_axis(curves[..., k], keepidxs, axis=1)
    #     for k in range(curves.shape[-1])
    # ], axis=-1)
    # return curves

@as_partial
def calculate_weights(data_bundle, scale=1.):
    curves = data_bundle['curves']
    weights = np.empty(curves.shape[:-1])
    batch_size = 1000
    print('Weight calculation started.')
    for num in range(0, curves.shape[0], batch_size):
        print(num+batch_size, end=' ')
        if ((num+batch_size)%10000 == 0):
            print()
        w_part = calculate_weighting_vectors(curves[num:num+batch_size]/scale)
        weights[num:num+batch_size] = w_part
    print()
    data_bundle['weights'] = weights
    return data_bundle

@as_partial
def calculate_max_dispers(data_bundle, scale=1.):
    curves = data_bundle['curves']
    weights = np.empty(curves.shape[:-1])
    batch_size = 1000
    print('Weight calculation started.')
    for num in range(0, curves.shape[0], batch_size):
        print(num+batch_size, end=' ')
        if ((num+batch_size)%10000 == 0):
            print()
        w_part = calculate_maximum_dispersion(curves[num:num+batch_size]/scale)
        weights[num:num+batch_size] = w_part
    print()
    data_bundle['max_dispers'] = weights
    return data_bundle

@as_partial
def extract_fourier(data_bundle, scale=1e0, n_filters=256, weight_type='w', random_state=42):
    curves = data_bundle['curves']
    if weight_type == 'm':
        weights = data_bundle['max_dispers']
    else:
        weights = data_bundle['weights']
    fourier_filter = FourierFilter(dim=curves.shape[-1], scale=scale, n_filters=n_filters, random_state=random_state)
    X = fourier_filter.apply(curves, weights, batch_size=256)
    data_bundle[f'X_fourier_{weight_type}'] = X
    return data_bundle


@as_partial
def extract_distance(data_bundle, scale=1e0, n_filters=256, random_state=42):
    curves = data_bundle['curves']
    # weights = data_bundle['weights']
    dist_filter = DistFilter(dim=curves.shape[-1], scale=scale, n_filters=n_filters, random_state=random_state)
    X = dist_filter.apply(curves, None, batch_size=256)
    data_bundle[f'X_dist'] = X
    return data_bundle


@as_partial
def split_into_train_test(data_bundle, test_ratio=0.2, ol_rate=0):
    seg_ids = data_bundle['seg_ids']
    idwise_cnt_normalized = np.sum([
        (seg_ids == s)*np.cumsum((seg_ids == s) / np.sum(seg_ids == s))
        for s in np.unique(seg_ids)
    ], axis=0)
    mask_train = idwise_cnt_normalized < (1-test_ratio)
    mask_test = idwise_cnt_normalized >= 1-test_ratio
    for s in np.unique(seg_ids):
        mask_id_s = seg_ids == s
        overlap_idx_min = np.min(np.where(mask_test&mask_id_s)[0])
        overlap_idxs = np.arange(overlap_idx_min, overlap_idx_min+1/(1-ol_rate)-1, dtype=np.int32)
        mask_test[overlap_idxs] = False
    data_bundle['mask_train'] = mask_train
    data_bundle['mask_test'] = mask_test
    # X, y = data_bundle['X'], data_bundle['seg_ids']
    # data_bundle['y'] = y
    # data_bundle['X_train'], data_bundle['X_test'] = X[mask_train], X[mask_test]
    # data_bundle['y_train'], data_bundle['y_test'] = y[mask_train], y[mask_test]
    return data_bundle


@as_partial
def load_cache(data_bundle, file_name):
    npz_loaded = np.load(file_name)
    for key in npz_loaded:
        data_bundle[key] = npz_loaded[key]
    return data_bundle


@as_partial
def save_features(data_bundle, file_name):
    np.savez(
        file_name, 
        X_dist=data_bundle['X_dist'],
        X_fourier_w=data_bundle['X_fourier_w']
    )
    return data_bundle


# def _make_geometric(segs, dim=3, lag=4, reduce=0, scale=1.):
#     embedder = Embedder(dim=dim, lag=lag, reduce=reduce)
#     length_loss = embedder.length_loss

#     curves = embedder.transform(segs)
#     weights = np.empty(curves.shape[:-1])
#     batch_size = 1000
#     for num in range(0, curves.shape[0], batch_size):
#         print(num+batch_size, end=' ')
#         if ((num+batch_size)%10000 == 0):
#             print()
#         w_part = calculate_weighting_vectors(curves[num:num+batch_size]/scale)[..., 0]
#         weights[num:num+batch_size] = w_part
#     return curves, weights