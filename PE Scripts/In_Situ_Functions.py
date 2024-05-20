import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import os, sys
from datetime import datetime
import skimage
import skimage.feature
import skimage.segmentation
import skimage.registration
import skimage.filters
import skimage.measure as sm
from skimage.transform import warp, SimilarityTransform
from csbdeep.utils import normalize
from glob import glob
from cellpose import models as cellpose_models
from cellpose import utils
from os import listdir
from os.path import isfile, join
import nd2reader
from natsort import natsorted
import In_Situ_Feldman as isfd
import warnings
warnings.filterwarnings('ignore')


def Manage_Save_Directories(_save_path):

    # list_subfolders = ['Reads', 'Reads_Amb', 'Nuclei', 'Cells', 'Report_Genotype', 'Cell_Genotype']
    # list_subfolders = ['Reads', 'Reads_Amb', 'Report_Genotype', 'Cell_Genotype']
    list_subfolders = ['Reads', 'Cell_Genotype']

    path_check = _save_path
    if not (os.path.isdir(path_check)):
        os.mkdir(path_check)

    for i in range(len(list_subfolders)):
        path_check = os.path.join(_save_path, list_subfolders[i])
        if not (os.path.isdir(path_check)):
            os.mkdir(path_check)

def Save(save_path, n_tile, n_well, nucs=0, cells=0, df_reads=0, df_reads_amb=0, df_report_genotype=0, df_cell_genotype=0):

    if not (isinstance(df_reads, int)):
        reads_save_path = save_path + '/Reads/Reads_tile_' + str(n_tile) + '_well_' + str(n_well) + '.csv'
        df_reads.to_csv(reads_save_path, index=False)

    if not (isinstance(df_reads_amb, int)):
        reads_amb_save_path = save_path + '/Reads_Amb/Reads_Amb_tile_' + str(n_tile) + '_well_' + str(n_well) + '.csv'
        df_reads_amb.to_csv(reads_amb_save_path, index=False)

    if not (isinstance(df_report_genotype, int)):
        reports_save_path = save_path + '/Report_Genotype/Report_Genotype_tile_' + str(n_tile) + '_well_' + str(n_well) + '.csv'
        df_report_genotype.to_csv(reports_save_path, index=False)

    if not (isinstance(df_cell_genotype, int)):
        cells_geno_save_path = save_path + '/Cell_Genotype/Cell_Genotype_tile_' + str(n_tile) + '_well_' + str(n_well) + '.csv'
        df_cell_genotype.to_csv(cells_geno_save_path, index=False)

    if not (isinstance(nucs, int)):
        nuc_save_path = save_path + '/Nuclei/Nuclei_tile_' + str(n_tile) + '_well_' + str(n_well) + '.npy'
        np.save(nuc_save_path, nucs)

    if not (isinstance(cells, int)):
        cell_save_path = save_path + '/Cells/Cells_tile_' + str(n_tile) + '_well_' + str(n_well) + '.npy'
        np.save(cell_save_path, cells)

def Library_Cycle_Min(_df_lib, col_name='sgRNA_seq'):

    N = len(_df_lib)

    for n_cyc in np.arange(1, 21):

        _df_lib['barcode'] = _df_lib[col_name].str.extract('(.{' + str(n_cyc) + '})')

        counts = len(_df_lib['barcode'].value_counts())

        print('Cycle:', n_cyc,' Non-unique barcodes:', N - counts)

def Concatenate_CSV(_path, _folder):
    _df_out = None

    report_path = join(_path, _folder)
    files = natsorted([f for f in listdir(report_path) if isfile(join(report_path, f)) and join(report_path, f).endswith('.csv')])

    if len(files) == 0:
        print('No files found')

    else:
        _df = pd.read_csv(join(report_path, files[0]))
        column_name = _df.columns.to_list()
        column_name = np.concatenate((['well', 'tile'], column_name))

        _data_out = np.empty([0, len(column_name)])
        _df_out = pd.DataFrame(data=_data_out, columns=column_name)

        for file in files:
            title = np.array(file.split('.')[0].split('_'))
            n_tile = int(title[int(np.squeeze(np.where(title == 'tile')[0]) + 1)])
            n_well = int(title[int(np.squeeze(np.where(title == 'well')[0]) + 1)])

            _df = pd.read_csv(join(report_path, file))
            _df.insert(0, 'tile', n_tile * np.ones([len(_df)], dtype=int) )
            _df.insert(0, 'well', n_well * np.ones([len(_df)], dtype=int))

            _df_out = pd.concat((_df_out, _df))

    return _df_out

def Plot_RGB(_red, _green, _blue):

    assert _red.shape == _green.shape and _red.shape == _blue.shape and _green.shape == _green.shape, 'Dimensions must match'

    h, w = _red.shape

    _red = (_red - _red.min()) / (_red.max() - _red.min())
    _green = (_green - _green.min()) / (_green.max() - _green.min())
    _blue = (_blue - _blue.min()) / (_blue.max() - _blue.min())

    _out = np.zeros([h,w,3])

    _out[:, :, 0] = _red
    _out[:, :, 1] = _green
    _out[:, :, 2] = _blue

    return _out

def Plot_GTAC_RGB(_img, bright=None):
    assert _img.ndim == 3

    _c, _h, _w = _img.shape

    if not (isinstance(bright, type(None))):
        for i in range(_c):
            _img[i] = Set_High_Q_exp(_img[i], bright)

    _out = np.zeros([_h, _w, 3])

    _out[:, :, 0] = 0.75 * _img[0] + 0.25 * _img[1]
    _out[:, :, 1] = 0.5 * _img[1] + 0.5 * _img[2]
    _out[:, :, 2] = 0.25 * _img[2] + 0.75 * _img[3]

    for i in range(3):
        _out[:, :, i] = (_out[:, :, i] - _out[:, :, i].min()) / (_out[:, :, i].max() - _out[:, :, i].min())

    return _out

    return _out

def Set_High_Q_exp(_img, _p):

    _q = 100*(1 - 10**-_p)

    flat = np.ravel(_img)
    P = np.percentile(flat, _q)
    _out = _img * (_img <= P) + P * (_img > P)

    return _out

def Tile_Number(_n):

    _out = 'None'

    if 0 <= _n and _n < 10:
        _out = '000' + str(_n)

    if 10 <= _n and _n < 100:
        _out = '00' + str(_n)

    if 100 <= _n and _n < 1000:
        _out = '0' + str(_n)

    if 1000 <= _n and _n < 10000:
        _out = str(_n)

    return _out


class InSitu:

    @staticmethod
    def Open_nd2(_path):
        return np.array(nd2reader.ND2Reader(_path), dtype=np.float64)

    @staticmethod
    def Import_ND2_by_Tile_and_Well(_n_tile, _n_well, _path_name):
        _data = np.empty([0])
        onlyfiles = [f for f in listdir(_path_name) if isfile(join(_path_name, f))]

        for i in range(len(onlyfiles)):

            if onlyfiles[i].endswith('.nd2'):

                n_tile = int(onlyfiles[i].split('_')[2])
                n_well = int(onlyfiles[i].split('_')[0].split('Well')[-1])

                if n_tile == _n_tile and n_well == _n_well:
                    full_path = _path_name + '/' + onlyfiles[i]
                    _data = np.array(nd2reader.ND2Reader(full_path), dtype=int)
                    only_name = onlyfiles[i].split('.')[0]
                    # print('Imported tile', n_tile, 'of well', n_well)

        if _data.shape[0] == np.array(0, dtype=tuple):
            print('Image of tile ' + str(_n_tile) + ' of well ' + str(_n_well) + ' not found')

        return _data  # , only_name

    @staticmethod
    def Assemble_Data_From_ND2(_n_tile, _n_well, _path_name, n_cycles=-1, verbose=False):

        """
        def compare_char(a, b):
            assert len(a) == len(b), "nd2 files should have the same char length"

            _char_comp = np.zeros([len(a)], dtype = 1)

            for i in range(len(a)):
                if a[i] != b[i]:
                    _char_comp[i] = 1

            return _char_comp
        """

        _data = np.empty([0])
        list_cycle_dirs = glob(_path_name + '/*')
        list_cycle_dirs = np.array(natsorted(list_cycle_dirs, key=lambda y: y.lower()), dtype=object)

        if n_cycles != -1:
            list_cycle_dirs = list_cycle_dirs[n_cycles]

        _complete_data = np.empty([len(list_cycle_dirs), 0, 0, 0])

        for j, cycle_path in enumerate(list_cycle_dirs):
            files = [f for f in listdir(cycle_path) if isfile(join(cycle_path, f)) and join(cycle_path, f).endswith('.nd2')]

            for i in range(len(files)):

                n_tile = int(files[i].split('_')[2])
                n_well = int(files[i].split('_')[0].split('Well')[-1])

                if n_tile == _n_tile and n_well == _n_well:
                    full_path = cycle_path + '/' + files[i]
                    _data_cycle = np.array(nd2reader.ND2Reader(full_path), dtype=np.float64)
                    if j == 0:
                        _complete_data = np.empty(np.concatenate(([len(list_cycle_dirs)], _data_cycle.shape)))
                        _complete_data[0] = _data_cycle
                    else:
                        _complete_data[j] = _data_cycle

                    if verbose:
                        print('Imported tile', n_tile, 'of well', n_well, cycle_path.split('/')[-1])

        return _complete_data

    @staticmethod
    def Find_Peaks(data, verbose=False):

        aligned = isfd.Snake._align_SBS(data, method='DAPI')
        if verbose:
            print('Aligned')

        loged = isfd.Snake._transform_log(aligned, skip_index=0)
        if verbose:
            print('Loged')

        maxed = isfd.Snake._max_filter(loged, 3, remove_index=0)
        if verbose:
            print('Maxed')

        std = isfd.Snake._compute_std(loged, remove_index=0)
        if verbose:
            print('Std')

        peaks = isfd.Snake._find_peaks(std)
        if verbose:
            print('Peaks')

        return maxed, peaks, aligned

    @staticmethod
    def Call_Bases(cells, maxed, peaks, THRESHOLD_STD, lim_low=0.25, lim_high=0.5):

        def dataframe_to_values(df, value='intensity'):
            cycles = df[CYCLE].value_counts()
            n_cycles = len(cycles)
            n_channels = len(df[CHANNEL].value_counts())
            x = np.array(df[value]).reshape(-1, n_cycles, n_channels)
            return x

        def transform_medians(X):
            arr = []
            for i in range(X.shape[1]):
                arr += [np.median(X[X.argmax(axis=1) == i], axis=0)]
            M = (np.array(arr)).T
            # M = M.T
            M = M / M.sum(axis=0)
            W = np.linalg.inv(M)
            Y = W.dot(X.T).T.astype(int)
            return Y, W

        """
        def Check_Q(_seq, _q, _idx, _lim_l=lim_low, _lim_h=lim_high):

            seq_list = list(_seq)
            for i in range(len(seq_list)):
                x1, x2 = _idx[i]
                if _q[i] < _lim_h:
                    seq_list[i] = 'N'
                    if _q[i] >= _lim_l:
                        if x1 + x2 == 1:
                            seq_list[i] = 'M'
                        if x1 + x2 == 5:
                            seq_list[i] = 'K'

            out = "".join(seq_list)

            return out
        
        def Find_X1_X2(_Y_):

            P, C, _ = _Y_.shape
            Idx = np.empty([P, C, 2], dtype=int)

            for p in range(P):
                y = _Y_[p]
                for c in range(C):
                    z = y[c]
                    idx_X1 = np.argmax(z)
                    z[idx_X1] = z.min() - 1
                    idx_X2 = np.argmax(z)
                    Idx[p, c, :] = idx_X1, idx_X2

            return Idx
        """

        cycles = list(range(1, maxed.shape[0] + 1))
        bases = ['G', 'T', 'A', 'C']
        CYCLE = 'cycle'
        CHANNEL = 'channel'
        POSITION_I = 'i'
        POSITION_J = 'j'
        INTENSITY = 'intensity'
        READ = 'read'
        CELL = 'cell'
        BARCODE = 'barcode'  # WELL='well'; TILE='tile'
        # PEAKS = 'peaks'

        if np.ndim(cells) == 3:
            cells = cells[-1]

        read_mask = (peaks > THRESHOLD_STD)
        values = maxed[:, :, read_mask].transpose([2, 0, 1])
        labels = cells[read_mask]
        positions = np.array(np.where(read_mask)).T

        index = (CYCLE, cycles), (CHANNEL, bases)
        names, levels = zip(*index)
        columns = pd.MultiIndex.from_product(levels, names=names)
        df = pd.DataFrame(values.reshape(values.shape[0], -1), columns=columns)

        df_positions = pd.DataFrame(positions, columns=[POSITION_I, POSITION_J])
        df_bases = (df.stack([CYCLE, CHANNEL])
                    .reset_index()
                    .rename(columns={0: INTENSITY, 'level_0': READ})
                    .join(pd.Series(labels, name=CELL), on=READ)
                    .join(df_positions, on=READ)
                    .sort_values([CELL, READ, CYCLE]))

        cycles = len(set(df_bases['cycle']))
        channels = len(set(df_bases['channel']))
        # df_bases.sort_values([WELL, TILE, CELL, READ, CYCLE, CHANNEL])
        df_bases.sort_values([CELL, READ, CYCLE, CHANNEL])

        X_ = dataframe_to_values(df_bases.query('cell > 0'))
        _, W = transform_medians(X_.reshape(-1, channels))
        X = dataframe_to_values(df_bases)
        Y = W.dot(X.reshape(-1, channels).T).T.astype(int)

        bases = sorted(set(df_bases[CHANNEL]))
        df_reads = df_bases.drop_duplicates([READ]).copy()
        Y_rearranged = Y.reshape(-1, cycles, channels)

        calls_int = Y_rearranged.argmax(axis=2)
        calls = np.array(list(bases))[calls_int]
        df_reads[BARCODE] = [''.join(x) for x in calls]

        Y_ = Y.reshape(-1, cycles, channels)
        Y_ = np.abs(np.sort(Y_, axis=-1).astype(float))
        Q = 1 - np.log(2 + Y_[..., -2]) / np.log(2 + Y_[..., -1])
        Q = (Q * 2).clip(0, 1)

        for i in range(len(Q[0])):
            df_reads['Q_%d' % i] = Q[:, i]

        df_reads = df_reads.drop([CYCLE, CHANNEL, INTENSITY], axis=1)

        # df_reads = df_reads.assign(Qumin=lambda x: x.filter(regex='Q_\d+').min(axis=1))
        # df_reads = df_reads.assign(Qumean=lambda x: x.filter(regex='Q_\d+').mean(axis=1))
        # df_reads = df_reads.assign(Qumax=lambda x: x.filter(regex='Q_\d+').max(axis=1))
        # print( df_reads['barcode'].str.extract('(.{3})'), df_reads['barcode'].str[1] )
        # print(df_reads.filter(regex='Q_\d+').to_numpy()[0])

        # Idx_X1_X2 = Find_X1_X2(Y_rearranged.copy())

        # df_reads_new = df_reads.copy()
        # for i in range(len(df_reads)):
        #     df_reads_new['barcode'].iloc[i] = Check_Q(df_reads['barcode'].iloc[i], Q[i], Idx_X1_X2[i])

        return df_reads

    @staticmethod
    def Assign_Simple_Ambiguity(_df_reads, lim=0.2):

        _df_reads_new = _df_reads.copy()

        for i in range(len(_df_reads)):
            barcode_str = list(_df_reads['barcode'].iloc[i])
            N_bar = _df_reads.filter(regex='Q_\d+').iloc[i].to_numpy() < lim
            for j in range(len(N_bar)):
                if N_bar[j]:
                    barcode_str[j] = 'N'

            _df_reads_new['barcode'].iloc[i] = ''.join(barcode_str)

        return _df_reads_new

    @staticmethod
    def Plot_Reads(cycle, calls_int, df_reads, nuclei, cells):

        marker_size = 5

        x = np.empty([0])
        y = np.empty([0])
        b = np.empty([0])
        for i in range(len(calls_int)):
            b_i = calls_int[i, cycle - 1]
            x_i = int(df_reads['j'].values[i])
            y_i = int(df_reads['i'].values[i])

            b = np.append(b, b_i)
            x = np.append(x, x_i)
            y = np.append(y, y_i)

        ind_A = np.where(b == 0)[0]
        ind_C = np.where(b == 1)[0]
        ind_G = np.where(b == 2)[0]
        ind_T = np.where(b == 3)[0]

        title_name = 'Cycle: ' + str(cycle)
        plt.title(title_name)

        plt.scatter(x[ind_A], y[ind_A], color='blue', s=marker_size, marker='.')
        plt.scatter(x[ind_C], y[ind_C], color='yellow', s=marker_size, marker='.')
        plt.scatter(x[ind_G], y[ind_G], color='red', s=marker_size, marker='.')
        plt.scatter(x[ind_T], y[ind_T], color='green', s=marker_size, marker='.')

        plt.imshow(2 * nuclei + cells, cmap='Greys')

    @staticmethod
    def Translate(_di, _dj, _W, _T, _M, _path):

        _M = np.pad(_M, ((1, 1), (1, 1)), 'constant', constant_values=-1)

        _img = InSitu.Import_ND2_by_Tile_and_Well(_T, _W, _path)

        _a = -2
        _b = -2
        _c = -2

        _I, _J = np.squeeze(np.where(_M == _T))

        transformed = np.zeros(_img.shape)
        for i in range(len(transformed)):
            transformed[i] = warp(_img[i], SimilarityTransform(translation=(-_dj, -_di)), preserve_range=True)

        if _di > 0 and _dj < 0:

            _a = _M[_I, _J + 1]
            _b = _M[_I - 1, _J + 1]
            _c = _M[_I - 1, _J]

            if _a == -1:
                img_J = np.zeros(transformed.shape)
            else:
                img_J = InSitu.Import_ND2_by_Tile_and_Well(_a, _W, _path)

            if _b == -1:
                img_IJ = np.zeros(transformed.shape)
            else:
                img_IJ = InSitu.Import_ND2_by_Tile_and_Well(_b, _W, _path)

            if _c == -1:
                img_I = np.zeros(transformed.shape)
            else:
                img_I = InSitu.Import_ND2_by_Tile_and_Well(_c, _W, _path)

            transformed[:, _di:, _dj:] = img_J[:, :-_di, :-_dj]
            transformed[:, :_di, _dj:] = img_IJ[:, -_di:, :-_dj]
            transformed[:, :_di, :_dj] = img_I[:, -_di:, -_dj:]

        if _di < 0 and _dj < 0:

            _a = _M[_I, _J + 1]
            _b = _M[_I + 1, _J + 1]
            _c = _M[_I + 1, _J]

            if _a == -1:
                img_J = np.zeros(transformed.shape)
            else:
                img_J = InSitu.Import_ND2_by_Tile_and_Well(_a, _W, _path)

            if _b == -1:
                img_IJ = np.zeros(transformed.shape)
            else:
                img_IJ = InSitu.Import_ND2_by_Tile_and_Well(_b, _W, _path)

            if _c == -1:
                img_I = np.zeros(transformed.shape)
            else:
                img_I = InSitu.Import_ND2_by_Tile_and_Well(_c, _W, _path)

            transformed[:, :_di, _dj:] = img_J[:, -_di:, :-_dj]
            transformed[:, _di:, _dj:] = img_IJ[:, :-_di, :-_dj]
            transformed[:, _di:, :-_dj] = img_I[:, :-_di, -_dj:]

        if _di > 0 and _dj > 0:

            _a = _M[_I, _J - 1]
            _b = _M[_I - 1, _J - 1]
            _c = _M[_I - 1, _J]

            if _a == -1:
                img_J = np.zeros(transformed.shape)
            else:
                img_J = InSitu.Import_ND2_by_Tile_and_Well(_a, _W, _path)

            if _b == -1:
                img_IJ = np.zeros(transformed.shape)
            else:
                img_IJ = InSitu.Import_ND2_by_Tile_and_Well(_b, _W, _path)

            if _c == -1:
                img_I = np.zeros(transformed.shape)
            else:
                img_I = InSitu.Import_ND2_by_Tile_and_Well(_c, _W, _path)

            transformed[:, _di:, :_dj] = img_J[:, :-_di, -_dj:]
            transformed[:, :_di, :_dj] = img_IJ[:, -_di:, -_dj:]
            transformed[:, :_di, _dj:] = img_I[:, -_di:, :-_dj]

        if _di < 0 and _dj > 0:

            _a = _M[_I, _J - 1]
            _b = _M[_I + 1, _J - 1]
            _c = _M[_I + 1, _J]

            if _a == -1:
                img_J = np.zeros(transformed.shape)
            else:
                img_J = InSitu.Import_ND2_by_Tile_and_Well(_a, _W, _path)

            if _b == -1:
                img_IJ = np.zeros(transformed.shape)
            else:
                img_IJ = InSitu.Import_ND2_by_Tile_and_Well(_b, _W, _path)

            if _c == -1:
                img_I = np.zeros(transformed.shape)
            else:
                img_I = InSitu.Import_ND2_by_Tile_and_Well(_c, _W, _path)

            for i in range(len(transformed)):
                transformed[i, :_di, :_dj] = img_J[i, -_di:, -_dj:]
                transformed[i, _di:, :_dj] = img_IJ[i, :-_di, -_dj:]
                transformed[i, _di:, _dj:] = img_I[i, :-_di, :-_dj]

        if _di == 0 and _dj < 0:

            _a = _M[_I, _J + 1]

            if _a == -1:
                img_J = np.zeros(transformed.shape)
            else:
                img_J = InSitu.Import_ND2_by_Tile_and_Well(_a, _W, _path)

            transformed[:, :, _dj:] = img_J[:, :, :-_dj]

        if _di < 0 and _dj == 0:

            _c = _M[_I + 1, _J]

            if _c == -1:
                img_I = np.zeros(transformed.shape)
            else:
                img_I = InSitu.Import_ND2_by_Tile_and_Well(_c, _W, _path)

            transformed[:, _di:, :] = img_I[:, :-_di, :]

        if _di == 0 and _dj > 0:

            _a = _M[_I, _J - 1]

            if _a == -1:
                img_J = np.zeros(transformed.shape)
            else:
                img_J = InSitu.Import_ND2_by_Tile_and_Well(_a, _W, _path)

            transformed[:, :, :_dj] = img_J[:, :, -_dj:]

        if _di > 0 and _dj == 0:

            _c = _M[_I - 1, _J]

            if _c == -1:
                img_I = np.zeros(transformed.shape)
            else:
                img_I = InSitu.Import_ND2_by_Tile_and_Well(_c, _W, _path)

            transformed[:, :_di, :] = img_I[:, -_di:, :]

        return transformed

    @staticmethod
    def Adjust_Channel_Intensities(_maxed, Bins=100, verbose=False):

        def f(x, A, B):
            return A * x + B

        def Map_Intensity_Distribution(_b, _h, _slope_avg, _intrs_avg):
            _x = (_h - _intrs_avg) / _slope_avg
            _c = np.mean(_x - _b)

            return _c

        _maxed_out = np.zeros(_maxed.shape)

        channels = 4
        low_lim = 1
        up_lim = 5

        h_b_lists = np.zeros([channels, 2], dtype=object)
        params = np.zeros([channels, 2])
        for i in range(channels):
            flat = np.ravel(_maxed[i])
            h, b = np.histogram(flat, bins=Bins)
            b = b[1:] / 2 + b[:-1] / 2

            h = np.log10(h)
            b = np.log10(b)
            h[h < 0] = 0
            b[b < 0] = 0

            logic_h = (h >= low_lim) * (h <= up_lim)
            b = b[logic_h]
            h = h[logic_h]

            h_b_lists[i, 0] = b
            h_b_lists[i, 1] = h

            popt, _ = scipy.optimize.curve_fit(f, b, h)
            params[i, 0] = popt[0]
            params[i, 1] = popt[1]

        slope_avg = np.mean(params[:, 0])
        intrs_avg = np.mean(params[:, 1])

        c_list = np.zeros([channels])
        for i in range(channels):
            c_list[i] = Map_Intensity_Distribution(h_b_lists[i, 0], h_b_lists[i, 1], slope_avg, intrs_avg)

        if verbose:

            colors = ['blue', 'green', 'orange', 'red']
            title_names = ['G', 'T', 'A', 'C', 'Pre-Aligned', 'Aligned']

            for i in range(channels):
                plt.subplot(2, 3, i + 1)
                plt.title(title_names[i])
                plt.scatter(h_b_lists[i, 0], h_b_lists[i, 1], c=colors[i], s=2)
                plt.plot(h_b_lists[i, 0], params[i, 0] * h_b_lists[i, 0] + params[i, 1], color=colors[i])
                plt.plot((h_b_lists[i, 1] - intrs_avg) / slope_avg, h_b_lists[i, 1], color='black')
                plt.scatter(h_b_lists[i, 0] + c_list[i], h_b_lists[i, 1], color='purple', s=2)

            plt.subplot(2, 3, 5)
            plt.title(title_names[4])
            for i in range(channels):
                plt.scatter(h_b_lists[i, 0], h_b_lists[i, 1], c=colors[i], s=2)

            plt.subplot(2, 3, 6)
            plt.title(title_names[5])
            for i in range(channels):
                plt.scatter(h_b_lists[i, 0] + c_list[i], h_b_lists[i, 1], color=colors[i], s=2)

            plt.show()

        for i in range(channels):
            _maxed_out[i] = 10 ** c_list[i] * _maxed[i]

        return _maxed_out


class Segment:

    @staticmethod
    def extract_props(_nuclie, _cells):

        props_nuclie = sm.regionprops(_nuclie)
        props_cells = sm.regionprops(_cells)

        x_n = np.empty([len(props_nuclie)], dtype=int)
        y_n = np.empty([len(props_nuclie)], dtype=int)
        x_c = np.empty([len(props_cells)], dtype=int)
        y_c = np.empty([len(props_cells)], dtype=int)
        for i in range(len(props_nuclie)):
            x_n[i], y_n[i] = props_nuclie[i].centroid
        for i in range(len(props_cells)):
            x_c[i], y_c[i] = props_cells[i].centroid

        L_list = np.empty([0, 4], dtype=int)
        for i in range(len(props_nuclie)):

            I = int(x_n[i])
            J = int(y_n[i])

            L = _cells[I, J]
            L_row = [[i + 1, L, I, J]]

            if L != 0:
                L_list = np.append(L_list, L_row, axis=0)

        _df = pd.DataFrame(L_list, columns=['nuc', 'cell', 'I', 'J'])

        return _df

    @staticmethod
    def Segment_Nuclei(_img, nuc_diameter=None, GPU=False):

        assert _img.ndim == 2, 'Input one channel for cell segmentation. Array should have 2 dimensions.'

        nuclei_model = cellpose_models.Cellpose(gpu=GPU, model_type='nuclei')
        # cyto_model = cellpose_models.Cellpose(gpu=False, model_type='cyto', torch=True)
        # cyto2_model = cellpose_models.Cellpose(gpu=False, model_type='cyto2', torch=True)

        _masks_nuc, _, _, _ = nuclei_model.eval(_img, diameter=nuc_diameter, channels=[[0, 0]])

        return _masks_nuc

    @staticmethod
    def Segment_Cells(_data, NUC=0, cell_diameter=None, GPU=False):

        # cell_model = cellpose_models.Cellpose(model_type='nuclei')
        # cell_model = cellpose_models.Cellpose(gpu=GPU, model_type='cyto') #, torch=True)
        cell_model = cellpose_models.Cellpose(gpu=GPU, model_type='cyto2')  # , torch=True)

        if isinstance(NUC, int):

            _masks_cells, _, _, _ = cell_model.eval(_data, diameter=cell_diameter, channels=[[0, 0]])

        else:

            h, w = _data.shape
            seg_data = np.zeros([h, w, 3])
            seg_data[:, :, 0] = _data
            seg_data[:, :, 2] = NUC

            _masks_cells, _, _, _ = cell_model.eval(seg_data, diameter=cell_diameter, channels=[[1, 3]])

        return _masks_cells

    @staticmethod
    def Plot_Segmented(img_nuc, img_cells, nuc, cells, choose='Combined', outline_size=0.001, color_nuc = 'lawngreen', color_cell = 'red'):

        if choose == 'Nuclei':
            outlines_nuc = utils.masks_to_outlines(nuc)
            outX_nuc, outY_nuc = np.nonzero(outlines_nuc)

            plt.subplot(1, 2, 1)
            plt.title('DAPI')
            plt.axis('off')
            plt.imshow(img_nuc)

            plt.subplot(1, 2, 2)
            plt.title('Nuclei Segmented')
            plt.axis('off')
            plt.imshow(img_nuc)
            plt.scatter(outY_nuc, outX_nuc, s=outline_size, c=color_cell)

        if choose == 'Cells':
            outlines_cells = utils.masks_to_outlines(cells)
            outX_cells, outY_cells = np.nonzero(outlines_cells)

            plt.subplot(1, 2, 1)
            plt.title('Syto-12')
            plt.axis('off')
            plt.imshow(cells)

            plt.subplot(1, 2, 2)
            plt.title('Cells Segmented')
            plt.axis('off')
            plt.imshow(img_nuc)
            plt.scatter(outY_cells, outX_cells, s=outline_size, c=color_cell)

        if choose == 'Segmented':
            outlines_nuc = utils.masks_to_outlines(nuc)
            outX_nuc, outY_nuc = np.nonzero(outlines_nuc)
            outlines_cells = utils.masks_to_outlines(cells)
            outX_cells, outY_cells = np.nonzero(outlines_cells)

            plt.subplot(1, 2, 1)
            plt.title('Nuclei Segmented')
            plt.axis('off')
            plt.imshow(img_cells)
            plt.scatter(outY_nuc, outX_nuc, s=outline_size, c=color_cell)

            plt.subplot(1, 2, 2)
            plt.title('Cells Segmented')
            plt.axis('off')
            plt.imshow(img_cells)
            plt.scatter(outY_cells, outX_cells, s=outline_size, c=color_cell)

        if choose == 'Combined':
            outlines_nuc = utils.masks_to_outlines(nuc)
            outX_nuc, outY_nuc = np.nonzero(outlines_nuc)
            outlines_cells = utils.masks_to_outlines(cells)
            outX_cells, outY_cells = np.nonzero(outlines_cells)

            plt.imshow(img_cells, cmap='Greys_r')
            plt.scatter(outY_nuc, outX_nuc, s=outline_size, c=color_nuc)
            plt.scatter(outY_cells, outX_cells, s=outline_size, c=color_cell)

        plt.show()

    @staticmethod
    def Label_and_Clean(_nuclie, _cells, save_pre_clean=True):

        df = Segment.extract_props(_nuclie, _cells)

        df2 = df.groupby('cell').filter(lambda x: len(x) > 1)
        df2 = df2.reset_index()
        for i in range(len(df2)):
            erase = df2.loc[i, 'nuc']
            _nuclie = (_nuclie != erase) * _nuclie

        df1 = df.groupby('cell').filter(lambda x: len(x) == 1)
        df1 = df1.reset_index()

        nuclie_new = np.zeros(_nuclie.shape)
        cells_new = np.zeros(_cells.shape)
        for i in range(len(df1)):
            new_label = -1*(i+1)
            old_label_cell = df1.loc[i, 'cell']
            old_label_nuc = df1.loc[i, 'nuc']
            df1.loc[i, 'cell'] = new_label
            df1.loc[i, 'nuc'] = new_label
            cells_new = (_cells == old_label_cell) * new_label + cells_new
            nuclie_new = (_nuclie == old_label_nuc) * new_label + nuclie_new

        pd.set_option("display.max_rows", None, "display.max_columns", None)

        frame = 3
        frame_1 = int(frame + 1)
        frame_2 = int(-1*frame)
        edge_1 = np.unique(cells_new[:,:frame_1])
        edge_2 = np.unique(cells_new[:, frame_2:])
        edge_3 = np.unique(cells_new[:frame_1, :])
        edge_4 = np.unique(cells_new[frame_2:, :])
        edge_5 = np.unique(nuclie_new[:,:frame_1])
        edge_6 = np.unique(nuclie_new[:, frame_2:])
        edge_7 = np.unique(nuclie_new[:frame_1, :])
        edge_8 = np.unique(nuclie_new[frame_2:, :])
        edge = np.unique(np.concatenate((edge_1, edge_2, edge_3, edge_4, edge_5, edge_6, edge_7, edge_8), axis=0))
        edge = edge[edge < 0]
        for j in edge:
            cells_new = (cells_new != j) * cells_new
            nuclie_new = (nuclie_new != j) * nuclie_new

        sorted_cells = np.unique(cells_new)
        sorted_cells = sorted_cells[sorted_cells < 0]

        nuclie_new_2 = np.zeros(_nuclie.shape)
        cells_new_2 = np.zeros(_cells.shape)
        for i in range(len(sorted_cells)):
            new_label = (i + 1)
            old_label_cell = sorted_cells[i]
            old_label_nuc = sorted_cells[i]
            cells_new_2 = (cells_new == old_label_cell) * new_label + cells_new_2
            nuclie_new_2 = (nuclie_new == old_label_nuc) * new_label + nuclie_new_2

        if save_pre_clean:
            nuc_out = np.concatenate(([_nuclie.astype(int)], [nuclie_new_2.astype(int)]), dtype=int)
            cell_out = np.concatenate(([_cells.astype(int)], [cells_new_2.astype(int)]), dtype=int)
        else:
            nuc_out = nuclie_new_2.astype(int)
            cell_out = cells_new_2.astype(int)

        return nuc_out, cell_out


class QC:

    @staticmethod
    def adjust_to(ary, n_top):

        # Takes the n_top-1 first values in an array,
        # And then sums the rest of the array for the n_top to end.

        ary = ary.to_numpy()
        if len(ary) >= n_top:
            temp = np.sum(ary[n_top:])
            ary = ary[:n_top]
            ary[-1] = temp
        else:
            temp = np.zeros([n_top])
            temp[0:len(ary)] = ary
            ary = temp
        return ary

    @staticmethod
    def count_gRNAs(_reads, n_top, Sum_End=False):

        column_name = 'sgRNA'

        list_gRNA = set(_reads[column_name])

        # print('Number of gRNAs: ', len(list_gRNA))

        list_barcodes = _reads[column_name].value_counts()


        # print(list_barcodes[:n_top])

        if Sum_End:
            temp = QC.adjust_to(list_barcodes, n_top)
            index_list = np.append(list_barcodes[:n_top - 1].index.to_numpy(), 'Others')
            list_barcodes = pd.DataFrame(data=temp, columns=[column_name], index=index_list)
            print(list_barcodes)

        else:
            list_barcodes = list_barcodes[:n_top]

        return len(list_gRNA), list_barcodes

    @staticmethod
    def cell_read_stats(df_reads, n_top, Sum_End=True):

        """
        Calculate for every cell the number of peaks that it contains,
        and the number of unique barcodes it contains
        """

        cells_list = df_reads['cell']
        barcodes_list = df_reads['barcode']

        if len(cells_list) > 0:
            unique_cells = np.sort(np.unique(cells_list))
            if unique_cells[0] == 0:
                unique_cells = unique_cells[1:]

            num_bars_list = np.empty([0], dtype=int)
            num_unq_bars_list = np.empty([0], dtype=int)
            for i in unique_cells:
                ind = np.where(cells_list == i)[0]
                num_unq_bars = np.unique(barcodes_list[ind])

                num_bars_list = np.append(num_bars_list, len(ind))
                num_unq_bars_list = np.append(num_unq_bars_list, int(len(num_unq_bars)))

            df_num = pd.DataFrame(num_bars_list, columns=['peaks_in_cell'])
            df_unq = pd.DataFrame(num_unq_bars_list, columns=['unq_bars'])

            num = df_num['peaks_in_cell'].value_counts()
            unq = df_unq['unq_bars'].value_counts()

            if Sum_End:
                num = QC.adjust_to(num, n_top)
                unq = QC.adjust_to(unq, n_top)
            else:
                num = num[:n_top]
                unq = unq[:n_top]

        else:
            num = np.zeros([n_top])
            unq = np.zeros([n_top])

        return num, unq

    @staticmethod
    def Tile_QC(_n_tile, _n_well, _path, nuc_save_path, cell_save_path, n_top=5, verbose=False):

        def Add_Repeated_Num_Columns(_text, _n, start=0):

            text_list = np.empty([_n], dtype=object)

            for i, w in enumerate(np.arange(start, _n + start)):
                text_list[i] = _text + str(w)

            return text_list

        reads_save_path = _path + '/Reads/Reads_tile_' + str(_n_tile) + '_well_' + str(_n_well) + '.csv'
        # report_save_path = _path + '/Report_Genotype/Report_Genotype_tile_' + str(_n_tile) + '_well_' + str(_n_well) + '.csv'
        cell_genotype_save_path = _path + '/Cell_Genotype/Cell_Genotype_tile_' + str(_n_tile) + '_well_' + str(_n_well) + '.csv'

        # nuc_save_path = _path + '/Nuclei/Nuclei_tile_' + str(_n_tile) + '_well_' + str(_n_well) + '.npy'
        # cell_save_path = _path + '/Cells/Cells_tile_' + str(_n_tile) + '_well_' + str(_n_well) + '.npy'

        # _path_new = 'TLopt_10March2022_Old/genotyping_results/well_' + str(_n_well) + '_N_2'
        # nuc_save_path = _path_new + '/Nuclei/Nuclei_tile_' + str(_n_tile) + '_well_' + str(_n_well) + '.npy'
        # cell_save_path = _path_new + '/Cells/Cells_tile_' + str(_n_tile) + '_well_' + str(_n_well) + '.npy'

        # _df_report = pd.read_csv(report_save_path)
        _df_reads = pd.read_csv(reads_save_path)

        if len(_df_reads) > 0:

            _df_cell_genotype = pd.read_csv(cell_genotype_save_path)
            _cells = np.load(os.path.join(cell_save_path, 'Seg_Cells-Well_' + str(_n_well) + '_Tile_' + str(_n_tile) + '.npy'))
            _nucs = np.load(os.path.join(nuc_save_path, 'Seg_Nuc-Well_' + str(_n_well) + '_Tile_' + str(_n_tile) + '.npy'))

            if np.ndim(_nucs) == 2:
                _nucs_pre = _nucs
                _nucs_post = _nucs
                if verbose:
                    print('Nuclei is only 2D array, pre and post cleanup will have the same stats')
                if np.ndim(_nucs) == 3:
                    _nucs_pre = _nucs[0]
                    _nucs_post = _nucs[1]

                if np.ndim(_cells) == 2:
                    _cells_pre = _cells
                    _cells_post = _cells
                    if verbose:
                        print('Cells is only 2D array, pre and post cleanup will have the same stats')
                if np.ndim(_cells) == 3:
                    _cells_pre = _cells[0]
                    _cells_post = _cells[1]

                _nucs_pre = np.ravel(_nucs_pre)
                _cells_pre = np.ravel(_cells_pre)
                _nucs_post = np.ravel(_nucs_post)
                _cells_post = np.ravel(_cells_post)

                qualities = _df_reads.filter(regex='Q_\d+').to_numpy()
                _, n_cyc = qualities.shape

                qc_path = os.path.join(_path, 'QC_Results.csv')

                column_names = np.array(
                    ['Tile', 'Well', 'Peaks', 'Peaks Outside Cells', 'Nuclei (Pre-clean)',
                     'Cells (Pre-clean)', 'Nuclei', 'Cells', 'Cells With Reads',
                     'Unique gRNAs', 'Total Nucleotides', 'A', 'C', 'G', 'T', 'K', 'M', 'N',
                     'Cell Attempted Match', 'Cell Matched', 'Cells Without Match', 'Cells With Ambiguous Reads'],
                    dtype=object)

                Repeated_Names = ['Avg Q Cyc ', 'Abundent gRNA ', 'Freq of gRNA ', 'Total Reads in Cell is ',
                                  'Unique Reads in Cell is ']

                if os.path.isfile(qc_path):

                    df_qc = pd.read_csv(qc_path)

                    column_names = df_qc.columns

                else:

                    add_list = Add_Repeated_Num_Columns(Repeated_Names[0], n_cyc)
                    column_names = np.concatenate((column_names, add_list))

                    add_list = Add_Repeated_Num_Columns(Repeated_Names[1], n_top, start=1)
                    column_names = np.concatenate((column_names, add_list))

                    add_list = Add_Repeated_Num_Columns(Repeated_Names[2], n_top, start=1)
                    column_names = np.concatenate((column_names, add_list))

                    add_list = Add_Repeated_Num_Columns(Repeated_Names[3], n_top, start=1)
                    column_names = np.concatenate((column_names, add_list))

                    add_list = Add_Repeated_Num_Columns(Repeated_Names[4], n_top, start=1)
                    column_names = np.concatenate((column_names, add_list))

                    df_qc = pd.DataFrame(data=np.empty([0, len(column_names)]), columns=column_names)

                df_qc_row = pd.DataFrame(data=np.empty([1, len(column_names)]), columns=column_names)

                df_qc_row[column_names[0]] = int(_n_tile)
                df_qc_row[column_names[1]] = int(_n_well)
                df_qc_row[column_names[2]] = len(_df_reads)
                df_qc_row[column_names[3]] = sum(_df_reads['cell'] == 0)

                df_qc_row[column_names[4]] = len(np.unique(_nucs_pre[_nucs_pre > 0]))
                df_qc_row[column_names[5]] = len(np.unique(_cells_pre[_cells_pre > 0]))
                df_qc_row[column_names[6]] = len(np.unique(_nucs_post[_nucs_post > 0]))
                df_qc_row[column_names[7]] = len(np.unique(_cells_post[_cells_post > 0]))

                cells_list = _df_reads['cell']
                num, unq = QC.cell_read_stats(_df_reads, n_top)
                unq_gRNA, list_barcodes = QC.count_gRNAs(_df_cell_genotype, n_top)
                df_qc_row[column_names[8]] = len(np.unique(cells_list[cells_list > 0]))
                df_qc_row[column_names[9]] = unq_gRNA

                data = ''.join(_df_reads['barcode'].to_list())
                split_data = np.array([char for char in data])
                df_qc_row[column_names[10]] = len(split_data)
                df_qc_row[column_names[11]] = sum(split_data == 'A')
                df_qc_row[column_names[12]] = sum(split_data == 'C')
                df_qc_row[column_names[13]] = sum(split_data == 'G')
                df_qc_row[column_names[14]] = sum(split_data == 'T')
                df_qc_row[column_names[15]] = sum(split_data == 'K')
                df_qc_row[column_names[16]] = sum(split_data == 'M')
                df_qc_row[column_names[17]] = sum(split_data == 'N')

                df_qc_row[column_names[18]] = 0 # len(_df_report)
                df_qc_row[column_names[19]] = 0 # len(_df_report[_df_report['assigned'] == True])
                df_qc_row[column_names[20]] = 0 # len(_df_report[_df_report['comment_2'] > 0])
                df_qc_row[column_names[21]] = 0 # len(_df_report[_df_report['comment_1'] > 0])

                for i in range(n_cyc):
                    name = Repeated_Names[0] + str(i)
                    df_qc_row[name] = np.round(np.mean(qualities[:, i]), decimals=3)

                for i in range(n_top):
                    name = Repeated_Names[1] + str(i + 1)
                    try:
                        df_qc_row[name] = list_barcodes.index[i]
                    except:
                        df_qc_row[name] = 'None'

                for i in range(n_top):
                    name = Repeated_Names[2] + str(i + 1)
                    try:
                        df_qc_row[name] = list_barcodes[i]
                    except:
                        df_qc_row[name] = 0

                for i in range(n_top):
                    name = Repeated_Names[3] + str(i + 1)
                    try:
                        df_qc_row[name] = int(num[i])
                    except:
                        df_qc_row[name] = 0

                for i in range(n_top):
                    name = Repeated_Names[4] + str(i + 1)
                    try:
                        df_qc_row[name] = int(unq[i])
                    except:
                        df_qc_row[name] = 0

                if sum(df_qc['Tile'] == _n_tile) > 0 and sum(df_qc['Well'] == _n_well) > 0:
                    df_qc = df_qc.drop(df_qc[ (df_qc['Tile'] == _n_tile) & (df_qc['Well'] == _n_well) ].index)

                df_qc = pd.concat([df_qc, df_qc_row], ignore_index=True, axis=0)
                df_qc = df_qc.sort_values(['Well', 'Tile'], ascending = (True, True))

                df_qc.to_csv(qc_path, index=False)

    @staticmethod
    def Quality_Score_Plots(_df_reads, _path, bins=50):

        _Q = _df_reads.filter(regex='Q_\d+').to_numpy()
        n_cyc = len(np.transpose(_Q))

        ax = 1
        min = _Q.min(axis=ax)
        max = _Q.max(axis=ax)
        avg = _Q.mean(axis=ax)
        std = _Q.std(axis=ax)

        plt.figure(figsize=(18, 8))

        plt.subplot(2, 4, 1)
        plt.title('Minimum per Peak')
        plt.hist(min, bins=bins)

        plt.subplot(2, 4, 2)
        plt.title('Maximum per Peak')
        plt.hist(max, bins=bins)

        plt.subplot(2, 4, 3)
        plt.title('Average per Peak')
        plt.hist(avg, bins=bins)

        plt.subplot(2, 4, 4)
        plt.title('Standard Deviation per Peak')
        plt.hist(std, bins=bins)

        plt.subplot(2, 1, 2)
        plt.title('Cycle Quality Distributions')
        plt.violinplot(_Q, showmeans=True)
        plt.xticks(range(1,n_cyc+1))
        plt.xlabel('Cycle')

        # plt.show()
        plt.savefig(join(_path, 'Quality_Score_Stats.png'))

    @staticmethod
    def Plot_Against_Bulk(_df_insitu, _df_lib):

        # Count occurences of every sgRNA
        count_insitu = _df_insitu['sgRNA'].value_counts().to_frame()
        ind_insitu = count_insitu.index.to_numpy()
        # Count occurences of every sgRNA
        count_lib = _df_lib['sgRNA'].value_counts().to_frame()
        ind_lib = count_lib.index.to_numpy()

        # Interscent Bulk and In-Situ libraries, use only sgRNAs that exhist in both
        ind_int = np.in1d(ind_insitu, ind_lib)

        # Extract sgRNAs that exhist in Bulk, setup x,y of plot
        hits = count_insitu.loc[ind_int]
        # Y_hits = np.squeeze(np.array(hits))
        # X_hits = np.arange(len(Y_hits))

        # Extract sgRNAs that do not exhist in Bulk, setup x,y of plot
        # no_hits = count_lib.loc[np.invert(ind_int)]
        # Y_no_hits = np.squeeze(np.array(no_hits))
        # X_no_hits = np.arange(len(no_hits))

        # Plots
        # plt.plot(np.log10(X_hits),np.log10(np.cumsum(Y_hits)),c='blue')
        # plt.plot(X_no_hits,np.cumsum(Y_no_hits),c='red')
        # plt.legend(['Mapped to Library','Not Mapped to Library'])
        # plt.xlabel('Rank')
        # plt.ylabel('Normalized Cumulative Frequency' )

        # print(2e6 * hits[-200:] / np.sum(hits))
        # print(len(hits))

        # plt.subplot(1,2,1)
        # YY = hits/np.sum(hits)
        # plt.plot(range(len(YY)), YY, c='blue')
        # plt.xlabel('Rank')
        # plt.ylabel('Fraction of Cells' )

        # plt.subplot(1,2,2)
        YY = 100 * hits / np.sum(hits)
        plt.plot(range(len(YY)), np.log10(YY), c='black')
        plt.xlabel('Rank')
        plt.ylabel('log10(Percent of Cells)')

        plt.show()

    @staticmethod
    def Plot_Rank(_df_insitu, _path, print_report=False):

        counts = _df_insitu['sgRNA'].value_counts()

        if print_report:
            print(counts)

        Y = counts.to_numpy()

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(Y)), np.cumsum(Y/np.sum(Y)), '.-')
        plt.xlabel('Rank')
        plt.ylabel('Cummulative Fraction')
        # plt.show()
        plt.savefig(join(_path, 'sgRNAs_by_Rank.png'))

    @staticmethod
    def Plot_sgRNAs_In_Gene(_df_insitu, _path):

        # Create gene column
        _df_insitu['gene'] = _df_insitu['sgRNA'].str.split('_', expand=True)[0].to_numpy()

        # Create an array containing only unique gene names
        unique_genes = np.unique(_df_insitu['gene'])

        # For every unique gene name, count the number of sgRNA that are associated with it
        sgRNA_counts = np.empty([len(unique_genes)], dtype=int)
        for i in range(len(unique_genes)):
            temp = _df_insitu[_df_insitu['gene'] == unique_genes[i]]
            sgRNA_counts[i] = len(np.unique(temp['sgRNA'].to_numpy()))

        # Arrange histrogram, and plot as a bar graph
        n_b = np.max(sgRNA_counts)
        h, b = np.histogram(sgRNA_counts, bins=n_b)

        plt.figure(figsize=(16, 8))
        plt.bar(range(1, n_b + 1), h)
        plt.xticks(range(1, n_b + 1))
        plt.xlabel('Number of sgRNAs in gene detected')
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig(join(_path, 'sgRNAs_in_Gene.png'))

    @staticmethod
    def Plot_sgRNAs_In_Intron(_df_insitu, _path):

        # Create gene column
        _df_insitu['gene'] = _df_insitu['sgRNA'].str.split('_', expand=True)[0].to_numpy()

        # Create an array containing only unique gene names
        unique_genes = np.unique(_df_insitu['gene'].to_numpy())

        # For every unique gene name, count the number of introns found
        sgRNA_counts = np.empty([len(unique_genes)])
        introns_list = np.empty([0])
        for i in range(len(unique_genes)):

            # Isolate unique sgRNA for every gene
            temp = _df_insitu[_df_insitu['gene'] == unique_genes[i]]
            sgRNA_names = np.unique(temp['sgRNA'].to_numpy())

            # For all unique sgRNAs in gene, extract intron information
            for j in range(len(sgRNA_names)):

                introns_numbers = sgRNA_names[j].split("_")

                # If statement is designed to handle both targeting and NTC intron notation, which are different
                if len(introns_numbers) == 1:
                    intron = introns_numbers[0]
                else:
                    intron = str(unique_genes[i]) + '_' + str(introns_numbers[1])

                introns_list = np.append(introns_list, intron)

        # Assemble intron counts into dataframe, and count frequencies of occurence
        df_introns = pd.DataFrame(introns_list)
        intron_count = df_introns.value_counts().to_numpy()

        # Assemble histrogram and plot as bar graph
        plt.figure(figsize=(8, 6))
        n_b = np.max(intron_count)
        h, b = np.histogram(intron_count, bins=n_b)
        plt.bar(range(1, n_b + 1), h)
        plt.xticks(range(1, n_b + 1))
        plt.xlabel('Number of sgRNAs in intron detected')
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig(join(_path, 'sgRNAs_in_Intron.png'))

    @staticmethod
    def Plot_Introns_In_Gene(_df_insitu, _path):

        # Create gene column
        _df_insitu['gene'] = _df_insitu['sgRNA'].str.split('_', expand=True)[0].to_numpy()

        # Create an array containing only unique gene names
        unique_genes = np.unique(_df_insitu['gene'].to_numpy())

        # For every unique gene name, count the number of sgRNAs found
        sgRNA_counts = np.empty([len(unique_genes)])
        introns_list_per_gene = np.empty([0], dtype=int)
        recovered_introns_per_gene_list = np.empty([0], dtype=int)
        for i in range(len(unique_genes)):

            # Isolate unique sgRNA for every gene
            temp = _df_insitu[_df_insitu['gene'] == unique_genes[i]]
            sgRNA_names = np.unique(temp['sgRNA'].to_numpy())

            # For all unique sgRNAs in gene, extract intron information
            for j in range(len(sgRNA_names)):

                introns_numbers = sgRNA_names[j].split("_")

                # If statement is designed to handle both targeting and NTC intron notation, which are different
                if len(introns_numbers) == 1:
                    intron = introns_numbers[0]
                else:
                    intron = str(unique_genes[i]) + '_' + str(introns_numbers[1])

                introns_list_per_gene = np.append(introns_list_per_gene, intron)

            # Count number of unique introns found in gene, append to global array for all genes
            number_introns_recovered = len(np.unique(introns_list_per_gene))
            recovered_introns_per_gene_list = np.append(recovered_introns_per_gene_list, number_introns_recovered)
            introns_list_per_gene = np.empty([0])

        # Assemble histrogram and plot as bar graph
        plt.figure(figsize=(8, 6))
        n_b = np.max(recovered_introns_per_gene_list)
        h, b = np.histogram(recovered_introns_per_gene_list, bins=n_b)
        plt.bar(range(1, n_b + 1), h)
        plt.xticks(range(1, n_b + 1))
        plt.xlabel('Number of introns with at least one sgRNA in gene detected')
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig(join(_path, 'Intron_in_Gene.png'))

    @staticmethod
    def Print_QC_Report(_df_insitu, _df_lib, _path, verbose=False):

        def Drop_NTC(_df):
            # Add a NTC T/F Columns, require a 'gene' column

            Targeting = np.zeros([len(_df)], dtype=bool)
            for i in range(len(_df)):
                try:
                    int(_df['gene'].iloc[i])
                except:
                    Targeting[i] = True

            _df['Targeting'] = Targeting

            return _df[_df['Targeting'] == True]

        #----- General Recovery Stats -----#
        _df_insitu['gene'] = _df_insitu['sgRNA'].str.split('_', expand=True)[0].to_numpy()
        _df_lib['gene'] = _df_lib['sgRNA'].str.split('_', expand=True)[0].to_numpy()

        count_insitu = _df_insitu['sgRNA'].value_counts().to_frame()
        # ind_insitu = count_insitu.index.to_numpy()
        count_lib = _df_lib['sgRNA'].value_counts().to_frame()

        # n_genes = len(_df_insitu['gene'].value_counts())
        # tot_genes = len(_df_lib['gene'].value_counts())
        n_sgRNAs = len(_df_insitu['sgRNA'].value_counts())
        tot_sgRNAs = len(_df_lib['sgRNA'].value_counts())

        _df_insitu = Drop_NTC(_df_insitu)
        _df_lib = Drop_NTC(_df_lib)

        n_genes_targeting = len(_df_insitu['gene'].value_counts())
        tot_genes_targeting = len(_df_lib['gene'].value_counts())
        n_sgRNAs_targeting = len(_df_insitu['sgRNA'].value_counts())
        tot_sgRNAs_targeting = len(_df_lib['sgRNA'].value_counts())

        _df_lib['intron'] = _df_lib['sgRNA'].str.split('_', expand=True)[1].to_numpy()
        _df_insitu['intron'] = _df_insitu['sgRNA'].str.split('_', expand=True)[1].to_numpy()

        _df_lib['gene_intron'] = _df_lib['gene'] + _df_lib['intron']
        _df_insitu['gene_intron'] = _df_insitu['gene'] + _df_insitu['intron']

        tot_targeting_introns = len(_df_lib['gene_intron'].value_counts())
        n_targeting_introns = len(_df_insitu['gene_intron'].value_counts())

        df_QC = pd.read_csv(join(_path, 'QC_Results.csv'))

        n_peaks = sum(df_QC['Peaks'])
        n_peaks_out = sum(df_QC['Peaks Outside Cells'])
        n_cells = sum(df_QC['Cells'])
        n_cells_reads = sum(df_QC['Cells With Reads'])
        n_attempt = sum(df_QC['Cell Attempted Match'])
        n_matched = sum(df_QC['Cell Matched'])
        n_unmatched = n_attempt - n_matched
        n_ambig = sum(df_QC['Cells With Ambiguous Reads'])

        n_reads_in_cell_1 = sum(df_QC['Total Reads in Cell is 1'])
        n_reads_in_cell_2 = sum(df_QC['Total Reads in Cell is 2'])
        n_reads_in_cell_3 = sum(df_QC['Total Reads in Cell is 3'])
        n_reads_in_cell_4 = sum(df_QC['Total Reads in Cell is 4'])
        n_reads_in_cell_5 = sum(df_QC['Total Reads in Cell is 5'])

        n_unique_reads_in_cell_1 = sum(df_QC['Unique Reads in Cell is 1'])
        n_unique_reads_in_cell_2 = sum(df_QC['Unique Reads in Cell is 2'])
        n_unique_reads_in_cell_3 = sum(df_QC['Unique Reads in Cell is 3'])
        n_unique_reads_in_cell_4 = sum(df_QC['Unique Reads in Cell is 4'])
        n_unique_reads_in_cell_5 = sum(df_QC['Unique Reads in Cell is 5'])

        n_nucs = sum(df_QC['Total Nucleotides'])
        n_A = sum(df_QC['A'])
        n_C = sum(df_QC['C'])
        n_G = sum(df_QC['G'])
        n_T = sum(df_QC['T'])
        n_K = sum(df_QC['K'])
        n_M = sum(df_QC['M'])
        n_N = sum(df_QC['N'])

        targeting_genes_recovered_percent = np.around(100 * n_genes_targeting / tot_genes_targeting, 1)
        targeting_introns_recovered_percent = np.around(100 * n_targeting_introns / tot_targeting_introns, 1)
        sgRNAs_recovered_percent = np.around(100 * n_sgRNAs / tot_sgRNAs, 1)
        targeting_sgRNAs_recovered_percent = np.around(100 * n_sgRNAs_targeting / tot_sgRNAs_targeting, 1)
        tot_NTC_sgRNAs = tot_sgRNAs - tot_sgRNAs_targeting
        n_NTC_sgRNA = n_sgRNAs - n_sgRNAs_targeting
        if tot_NTC_sgRNAs != 0:
            NTC_sgRNAs_recovered_percent = np.around(100 * n_NTC_sgRNA / tot_NTC_sgRNAs, 1)
        else:
            NTC_sgRNAs_recovered_percent = 0
        n_peaks_in = n_peaks - n_peaks_out
        peaks_inside_cells_percent = np.around(100 * n_peaks_in / n_peaks, 1)
        cells_with_reads_found_percent = np.around(100 * n_cells_reads / n_cells, 1)

        n_reads_in_cell_1_percent = np.around(100 * n_reads_in_cell_1 / n_cells_reads, 1)
        n_reads_in_cell_2_percent = np.around(100 * n_reads_in_cell_2 / n_cells_reads, 1)
        n_reads_in_cell_3_percent = np.around(100 * n_reads_in_cell_3 / n_cells_reads, 1)
        n_reads_in_cell_4_percent = np.around(100 * n_reads_in_cell_4 / n_cells_reads, 1)
        n_reads_in_cell_5_percent = np.around(100 * n_reads_in_cell_5 / n_cells_reads, 1)

        cells_with_1_unique_read_percent = np.around(100 * n_reads_in_cell_1 / n_cells_reads, 1)
        cells_with_2_unique_read_percent = np.around(100 * n_reads_in_cell_2 / n_cells_reads, 1)
        cells_with_3_unique_read_percent = np.around(100 * n_reads_in_cell_3 / n_cells_reads, 1)
        cells_with_4_unique_read_percent = np.around(100 * n_reads_in_cell_4 / n_cells_reads, 1)
        cells_with_5_unique_read_percent = np.around(100 * n_reads_in_cell_5 / n_cells_reads, 1)

        n_attempt_percent = np.around(100 * n_attempt / n_cells_reads, 1)
        n_matched_percent = np.around(100 * n_matched / n_cells_reads, 1)
        n_unmatched_percent = np.around(100 * n_unmatched / n_cells_reads, 1)
        n_ambig_percent = np.around(100 * n_ambig / n_cells_reads, 1)

        A_percent = np.around(100 * n_A / n_nucs, 1)
        C_percent = np.around(100 * n_C / n_nucs, 1)
        G_percent = np.around(100 * n_G / n_nucs, 1)
        T_percent = np.around(100 * n_T / n_nucs, 1)
        K_percent = np.around(100 * n_K / n_nucs, 1)
        M_percent = np.around(100 * n_M / n_nucs, 1)
        N_percent = np.around(100 * n_N / n_nucs, 1)

        QC_file = open(join(_path, 'QC_Report.txt'), 'w+')

        QC_file.write('|-------------------------- In-Situ Sequencing QC Report --------------------------|' + '\n')
        QC_file.write('Report date and time: ' + str(datetime.now()) + '\n\n')

        QC_file.write('Tiles analyzed: ' + str(len(df_QC)) + '\n')
        QC_file.write('Wells involved in analysis: ' + str(len(df_QC['Well'].value_counts())) + '\n\n')

        QC_file.write('---------- Library Stats ----------' + '\n')
        QC_file.write('Total targeting genes in library: ' + str(tot_genes_targeting) + '\n')
        QC_file.write('Total introns targeted in library: ' + str(tot_targeting_introns) + '\n')
        QC_file.write('Total sgRNAs in library: ' + str(tot_sgRNAs) + '\n')
        QC_file.write('Total targeting sgRNAs in library: ' + str(tot_sgRNAs_targeting) + '\n')
        QC_file.write('Total NTC sgRNAs in library: ' + str(tot_NTC_sgRNAs) + '\n\n')

        QC_file.write('---------- Recovery Stats ----------' + '\n')
        QC_file.write('Targeting genes recovered: ' + str(n_genes_targeting) + ' (' + str(targeting_genes_recovered_percent) + '%)' + '\n')
        QC_file.write('Targeting introns recovered: ' + str(n_targeting_introns) + ' (' + str(targeting_introns_recovered_percent) + '%)' + '\n')
        QC_file.write('sgRNAs recovered: ' + str(n_sgRNAs) + ' (' + str(sgRNAs_recovered_percent) + '%)' + '\n')
        QC_file.write('Targeting sgRNAs recovered: ' + str(n_sgRNAs_targeting) + ' (' + str(targeting_sgRNAs_recovered_percent) + '%)' + '\n')
        QC_file.write('NTC sgRNAs recovered: ' + str(n_NTC_sgRNA) + ' (' + str(NTC_sgRNAs_recovered_percent) + '%)' + '\n\n')

        QC_file.write('---------- Cell and Reads Stats ----------' + '\n')
        QC_file.write('Peaks found: ' + str(n_peaks) + '\n')
        QC_file.write('Peaks inside cells: ' + str(n_peaks_in) + ' (' + str(peaks_inside_cells_percent) + '%)' + '\n')
        QC_file.write('Cells found: ' + str(n_cells) + '\n')
        QC_file.write('Cells with reads found: ' + str(n_cells_reads) + ' (' + str(cells_with_reads_found_percent) + '% of found cells)' + '\n\n')

        QC_file.write('---------- Reads Frequency Stats ----------' + '\n')
        QC_file.write('Cells with 1 read: ' + str(n_reads_in_cell_1) + ' (' + str(n_reads_in_cell_1_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with 2 read: ' + str(n_reads_in_cell_2) + ' (' + str(n_reads_in_cell_2_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with 3 read: ' + str(n_reads_in_cell_3) + ' (' + str(n_reads_in_cell_3_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with 4 read: ' + str(n_reads_in_cell_4) + ' (' + str(n_reads_in_cell_4_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with 5 read: ' + str(n_reads_in_cell_5) + ' (' + str(n_reads_in_cell_5_percent) + '% of cells with reads)' + '\n\n')

        QC_file.write('---------- Reads Uniqueness Stats ----------' + '\n')
        QC_file.write('Cells with 1 unique read: ' + str(n_unique_reads_in_cell_1) + ' (' + str(cells_with_1_unique_read_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with 2 unique read: ' + str(n_unique_reads_in_cell_2) + ' (' + str(cells_with_2_unique_read_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with 3 unique read: ' + str(n_unique_reads_in_cell_3) + ' (' + str(cells_with_3_unique_read_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with 4 unique read: ' + str(n_unique_reads_in_cell_4) + ' (' + str(cells_with_4_unique_read_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with 5 unique read: ' + str(n_unique_reads_in_cell_5) + ' (' + str(cells_with_5_unique_read_percent) + '% of cells with reads)' + '\n\n')

        QC_file.write('---------- Genotype Mapping Stats ----------' + '\n')
        QC_file.write('Cells with high enough quality barcodes: ' + str(n_attempt) + ' (' + str(n_attempt_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with a match: ' + str(n_matched) + ' (' + str(n_matched_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with good barcode and without a match: ' + str(n_unmatched) + ' (' + str(n_unmatched_percent) + '% of cells with reads)' + '\n')
        QC_file.write('Cells with several matches and ambiguous (subgroup of unmatched): ' + str(n_ambig) + ' (' + str(n_ambig_percent) + '% of cells with reads)' + '\n\n')

        QC_file.write('---------- Nucleotide Stats ----------' + '\n')
        QC_file.write('Number of nucleotides sequenced: ' + str(n_nucs) + '\n')
        QC_file.write('Number of A sequenced: ' + str(n_A) + ' (' + str(A_percent) + '%)' + '\n')
        QC_file.write('Number of C sequenced: ' + str(n_C) + ' (' + str(C_percent) + '%)' + '\n')
        QC_file.write('Number of G sequenced: ' + str(n_G) + ' (' + str(G_percent) + '%)' + '\n')
        QC_file.write('Number of T sequenced: ' + str(n_T) + ' (' + str(T_percent) + '%)' + '\n')
        QC_file.write('Number of K sequenced: ' + str(n_K) + ' (' + str(K_percent) + '%)' + '\n')
        QC_file.write('Number of M sequenced: ' + str(n_M) + ' (' + str(M_percent) + '%)' + '\n')
        QC_file.write('Number of N sequenced: ' + str(n_N) + ' (' + str(N_percent) + '%)' + '\n')

        QC_file.close()

        if verbose:

            print('Tiles analyzed:', len(df_QC))
            print('Wells involved in analysis:', len(df_QC['Well'].value_counts()))

            print('Total targeting genes in library:', tot_genes_targeting)
            print('Total introns targeted in library:', tot_targeting_introns)
            print('Total sgRNAs in library:', tot_sgRNAs)
            print('Total targeting sgRNAs in library:', tot_sgRNAs_targeting)
            tot_NTC_sgRNAs = tot_sgRNAs - tot_sgRNAs_targeting
            print('Total NTC sgRNAs in library:', tot_NTC_sgRNAs)

            print('Targeting genes recovered:', n_genes_targeting, '  (', n_genes_targeting, '%)')
            print('Targeting introns recovered:', n_targeting_introns, '  (', targeting_genes_recovered_percent, '%)')
            print('sgRNAs recovered:', n_sgRNAs, '  (', sgRNAs_recovered_percent, '%)')
            print('Targeting sgRNAs recovered:', n_sgRNAs_targeting, '  (', targeting_sgRNAs_recovered_percent, '%)')
            n_NTC_sgRNA = n_sgRNAs - n_sgRNAs_targeting
            print('NTC sgRNAs recovered:', n_NTC_sgRNA, '  (', NTC_sgRNAs_recovered_percent, '%)')

            print('Peaks found:', n_peaks)
            print('Peaks inside cells:', n_peaks - n_peaks_out, '  (', peaks_inside_cells_percent, '%)')
            print('Cells found:',  n_cells)
            print('Cells with reads found:', n_cells_reads, '  (', cells_with_reads_found_percent, '% of found cells)')

            print('Cells with 1 read:', n_reads_in_cell_1, '  (', n_reads_in_cell_1_percent, '% of cells with reads)')
            print('Cells with 2 read:', n_reads_in_cell_2, '  (', n_reads_in_cell_1_percent, '% of cells with reads)')
            print('Cells with 3 read:', n_reads_in_cell_3, '  (', n_reads_in_cell_1_percent, '% of cells with reads)')
            print('Cells with 4 read:', n_reads_in_cell_4, '  (', n_reads_in_cell_1_percent, '% of cells with reads)')
            print('Cells with 5 read:', n_reads_in_cell_5, '  (', n_reads_in_cell_1_percent, '% of cells with reads)')

            print('Cells with 1 unique read:', n_unique_reads_in_cell_1, '  (', cells_with_1_unique_read_percent, '% of cells with reads)')
            print('Cells with 2 unique read:', n_unique_reads_in_cell_2, '  (', cells_with_2_unique_read_percent, '% of cells with reads)')
            print('Cells with 3 unique read:', n_unique_reads_in_cell_3, '  (', cells_with_3_unique_read_percent, '% of cells with reads)')
            print('Cells with 4 unique read:', n_unique_reads_in_cell_4, '  (', cells_with_4_unique_read_percent, '% of cells with reads)')
            print('Cells with 5 unique read:', n_unique_reads_in_cell_5, '  (', cells_with_5_unique_read_percent, '% of cells with reads)')

            print('Cells with high enough quality barcodes:', n_attempt, '  (', n_attempt_percent, '% of cells with reads)')
            print('Cells with a match:', n_matched, '  (', n_matched_percent, '% of cells with reads)')
            print('Cells with good barcode and without a match:', n_unmatched, '  (', n_unmatched_percent, '% of cells with reads)')
            print('Cells with several matches and ambiguous (subgroup of unmatched):', n_ambig, '  (', n_ambig_percent, '% of cells with reads)')

            print('Number of nucleotides sequenced:', n_nucs)
            print('Number of A sequenced:', n_A, '  (', A_percent, '%)')
            print('Number of C sequenced:', n_C, '  (', C_percent, '%)')
            print('Number of G sequenced:', n_G, '  (', G_percent, '%)')
            print('Number of T sequenced:', n_T, '  (', T_percent, '%)')
            print('Number of K sequenced:', n_K, '  (', K_percent, '%)')
            print('Number of M sequenced:', n_M, '  (', M_percent, '%)')
            print('Number of N sequenced:', n_N, '  (', N_percent, '%)')


class Lookup:

    @staticmethod
    def Reads_Q_lim(_df, _lim):
        _data_Q = _df[[i for i in _df.columns if i.startswith('Q')]].to_numpy()
        min_list = np.zeros([len(_data_Q)])
        for i in range(len(_data_Q)):
            min_list[i] = np.min(_data_Q[i, :])
        _df['min'] = min_list

        return _df[_df['min'] >= _lim]

    @staticmethod
    def Split_Nucleotides(_seq_list):

        _seq_list = np.array(_seq_list)

        if _seq_list.ndim == 0:
            _seq_list = np.array([_seq_list])

        if _seq_list.ndim == 2:
            _seq_list = np.squeeze(_seq_list)

        n_cyc = len(_seq_list[0])

        out = np.empty([len(_seq_list), n_cyc], dtype=object)

        for i, seq in enumerate(_seq_list):
            _split = np.array([char for char in seq])
            out[i] = _split

        return out

    @staticmethod
    def Choose_Barcodes(_df_reads, _df_lib, _nuclie, _cells, N_lim=None, verbose=False):

        def Count_Low_Q(_seq):
            bases = np.array(['A', 'C', 'G', 'T'])
            n = 0
            for c in list(_seq):
                if sum(c == bases) == 0:
                    n = n + 1
            return n

        def Choose_Barcode_L2(_seq, _choices):

            def ACGT_Mask(_seq):

                _seq = list(str(_seq))
                _mask = np.empty([len(_seq)], dtype=bool)
                for i, c in enumerate(_seq):
                    _mask[i] = np.sum(c == np.array(['A', 'C', 'G', 'T'])) > 0

                return _mask

            idx = np.arange(len(list(_seq)))
            mask = ACGT_Mask(_seq)
            drop_list = np.empty([0], dtype=int)
            for i in range(len(_choices)):

                seq_option = _choices['barcode'].iloc[i]
                _drop = False
                for n in idx[mask]:
                    if _seq[n] != seq_option[n]:
                        _drop = True

                for n in idx[np.invert(mask)]:
                    if _seq[n] == 'K':
                        if seq_option[n] == 'A' or seq_option[n] == 'C':
                            _drop = True

                    if _seq[n] == 'M':
                        if seq_option[n] == 'G' or seq_option[n] == 'T':
                            _drop = True

                if _drop:
                    drop_list = np.append(drop_list, _choices.index[i])

            _choices = _choices.drop(drop_list)

            return _choices

        def Choose_Barcode_L3_popular(_choices_L3):

            _choice_Final = None

            df_counts = _choices_L3['sgRNA'].value_counts()
            counts = df_counts.to_numpy()

            max_counts = np.sum(np.max(counts) == counts)

            if max_counts == 1:
                _choice_Final = _choices_L3[_choices_L3['sgRNA'] == df_counts.index[0]]

            return _choice_Final

        Q = _df_reads.filter(regex='Q_\d+').to_numpy()
        _, n_cyc = Q.shape
        if isinstance(N_lim, type(None)):
            N_lim = n_cyc

        _df_props = Segment.extract_props(_nuclie, _cells)

        # Drop reads outside cells
        _df_reads = _df_reads[_df_reads['cell'] > 0]

        # Create 'barcode' column, cut gRNA sequence to the number of cycles performed
        _df_lib['barcode'] = _df_lib['sgRNA_seq'].str[:n_cyc]
        lib_barcode = Lookup.Split_Nucleotides(_df_lib['barcode'].to_numpy())
        n_lib = len(_df_lib)

        # Tally the number of KMNs in barcode for each read
        # N_list = np.empty([len(_df_reads)])
        # for i in range(len(_df_reads)):
        #     N_list[i] = Count_Low_Q(_df_reads['barcode'].iloc[i])

        # Keep only reads with 'N_lim' 'N's or less
        # _df_reads_new = _df_reads[N_list <= N_lim]

        # ---Begin individual cell matching---#
        # Create list with remaining cell labels
        cell_list = np.unique(_df_reads['cell'].to_numpy(dtype=int))

        final_columns = ['sgRNA', 'cell', 'i_nuc', 'j_nuc', 'reads', 'matched', 'correct', 'mismatch_max', 'mismatch_min', 'barcode']
        df_out = pd.DataFrame(data=np.empty([0, len(final_columns)]), columns=final_columns)

        # Isolate individual cell dataframe
        for i, c in enumerate(cell_list):
            # Extract individual cell dataframe

            if verbose:
                print(i, 'Cell:', c, '###############################################')

            df_cell = _df_reads[_df_reads['cell'] == c]

            if verbose:
                print('Reads in cell')
                print(df_cell[['read', 'barcode']].reset_index(drop=True))

            choices_L3 = pd.DataFrame( data=np.empty([0, len(_df_lib.columns)]), columns=_df_lib.columns)
            for r, seq in zip(df_cell['read'], df_cell['barcode']):

                seq_mult = Lookup.Split_Nucleotides(n_lib * [seq])
                matches = np.sum(seq_mult == lib_barcode, axis=1)
                choices = _df_lib.iloc[np.flatnonzero(matches == matches.max())]
                choices_L2 = Choose_Barcode_L2(seq, choices)

                if len(choices_L2) == 0:
                    choices_L2 = pd.DataFrame(np.expand_dims(np.repeat(['No'], len(choices_L2.columns)), 0), columns=choices_L2.columns)

                choices_L2['mismatch'] = int(n_cyc - matches.max())
                choices_L2['query'] = seq
                choices_L2['read'] = r

                choices_L3 = pd.concat((choices_L3, choices_L2))

            choices_L3 = choices_L3[choices_L3['mismatch'] <= N_lim]

            if verbose:
                print('Match options')
                short_list = np.empty([len(choices_L3)], dtype=object)
                for i in range(len(choices_L3)):
                    try:
                        short_list[i] = choices_L3['sgRNA'].iloc[i].split('_')[2] + '_' + choices_L3['sgRNA'].iloc[i].split('_')[4]
                    except:
                        short_list[i] = choices_L3['sgRNA'].iloc[i]
                choices_L3['sgRNA_short'] = short_list
                print(choices_L3[['sgRNA_short', 'barcode', 'mismatch', 'query', 'read']].reset_index(drop=True))

            choices_L3 = choices_L3[choices_L3['barcode'] != 'No']
            if len(choices_L3) != 0:
                choice_final = Choose_Barcode_L3_popular(choices_L3)

                if not (isinstance(choice_final, type(None))):

                    if verbose:
                        print('-----------------------------------------------')
                        print('Final reads:', choice_final[['sgRNA_short', 'barcode', 'mismatch', 'query', 'read']])
                        print('-----------------------------------------------')
                        print('Final barcode:', choice_final['barcode'].value_counts())

                    _df_series = _df_props[_df_props['cell'] == c]
                    data_choice_final = np.array([
                        [choice_final['sgRNA'].iloc[0]],
                        [c],
                        [_df_series['I'].iloc[0]],
                        [_df_series['J'].iloc[0]],
                        [len(df_cell)],
                        [len(choice_final)],
                        [len(choices_L3)],
                        [int(choice_final['mismatch'].max())],
                        [int(choice_final['mismatch'].min())],
                        [choice_final['sgRNA_seq'].iloc[0]]], dtype=object)

                    df_choice_final = pd.DataFrame(data=np.transpose(data_choice_final), columns=final_columns)
                    df_out = pd.concat((df_out, df_choice_final), ignore_index=True)

                else:
                    if verbose:
                        print('Too many barcodes to choose from, ambiguous')
            else:
                if verbose:
                    print('No match found for any of cell barcodes')

        return df_out

    @staticmethod
    def Choose_Barcodes_Blast(_df_reads, _df_lib):

        Q = _df_reads.filter(regex='Q_\d+').to_numpy()
        _, n_cyc = Q.shape

        # Drop reads outside cells
        _df_reads = _df_reads[_df_reads['cell'] > 0]

        # Create 'barcode' column, cut gRNA sequence to the number of cycles performed
        _df_lib['barcode'] = _df_lib['sgRNA_seq'].str[:n_cyc]
        # lib_barcode = Split_Nucleotides(_df_lib['barcode'].to_numpy())
        lib_column_names = _df_lib.columns.to_numpy()
        n_lib = len(_df_lib)


class Optimize:

    @staticmethod
    def Plot_Mapping_vs_Threshold(_df_reads, _df_lib, threshold_var='peak', ax=None):

        _, n_cyc = _df_reads.filter(regex='Q_\d+').to_numpy().shape
        barcodes = _df_lib['sgRNA_seq'].str[:n_cyc].to_list()

        _df_summary = isfd.QC.plot_mapping_vs_threshold(_df_reads, barcodes, threshold_var=threshold_var, ax=ax)

        print(_df_summary)

        # return _df_summary

    @staticmethod
    def Compare_Assignment(_df_assigned_1, _df_assigned_2, title_1='df_1', title_2='df_2', verbose=False):

        cells_1 = _df_assigned_1['cell'].to_numpy()
        cells_2 = _df_assigned_2['cell'].to_numpy()

        cells = np.union1d(cells_1, cells_2)

        compare_data = np.empty([len(cells), 5], dtype=object)

        for i in range(len(cells)):

            df_sgRNA_1 = _df_assigned_1[_df_assigned_1['cell'] == cells[i]]
            if len(df_sgRNA_1) == 0:
                sgRNA_1 = 'None'
            else:
                sgRNA_1 = df_sgRNA_1['sgRNA'].to_list()[0]

            df_sgRNA_2 = _df_assigned_2[_df_assigned_2['cell'] == cells[i]]
            if len(df_sgRNA_2) == 0:
                sgRNA_2 = 'None'
            else:
                sgRNA_2 = df_sgRNA_2['sgRNA'].to_list()[0]

            compare_data[i, 0] = cells[i]
            compare_data[i, 1] = sgRNA_1
            compare_data[i, 2] = sgRNA_2
            compare_data[i, 3] = len(df_sgRNA_1) + len(df_sgRNA_2)
            compare_data[i, 4] = sgRNA_1 == sgRNA_2

        df_compare = pd.DataFrame(data=compare_data, columns=['cell', title_1, title_2, 'found', 'match'])

        # df_check = df_compare[df_compare['found'] == 2]
        # print(df_check[df_check['match'] == False])
        df_found = df_compare[df_compare['found'] == 2]
        match_index = df_found['match'].to_numpy()

        per = np.round(100 * sum(match_index) / len(match_index), 1)

        if verbose:
            print('Cells in both:', len(df_found), '  Cells with mapping disagreement:', len(df_found[df_found['match'] == False]), '  Percent:', per)

        return df_compare

    @staticmethod
    def Optimize_Peaks_Threshold(N_lim_list, thresholds_list, cells, maxed, peaks, _df_lib, lim_low=0.25, lim_high=0.5, verbose=True):

        recovered_cells = np.zeros([len(N_lim_list), len(thresholds_list)], dtype=int)
        recovered_sgRNAs = np.zeros([len(N_lim_list), len(thresholds_list)], dtype=int)
        for n, N_lim in enumerate(N_lim_list):

            for t, THRESHOLD_STD in enumerate(thresholds_list):

                _df_reads = InSitu.Call_Bases(cells, maxed, peaks, THRESHOLD_STD, lim_low=lim_low, lim_high=lim_high)

                peaks_list = np.empty([len(_df_reads)])
                for c in range(len(_df_reads)):
                    peaks_list[c] = peaks[_df_reads['i'].iloc[c], _df_reads['j'].iloc[c]]
                _df_reads['peaks'] = peaks_list

                df_assigned, _ = Lookup.Choose_Barcodes(_df_reads, _df_lib, N_lim=N_lim, verbose=False)

                recovered_cells[n, t] = len(df_assigned)
                recovered_sgRNAs[n, t] = len(df_assigned['sgRNA'].value_counts())

                if verbose:
                    print('Ambig:', N_lim, ' Thresh:', THRESHOLD_STD, ' Cells assigned:', len(df_assigned), ' sgRNA found:', len(df_assigned['sgRNA'].value_counts()))

        _df_recovered_cells = pd.DataFrame(data=recovered_cells, columns=N_lim_list, index=thresholds_list)
        _df_recovered_sgRNAs = pd.DataFrame(data=recovered_sgRNAs, columns=N_lim_list, index=thresholds_list)

        return _df_recovered_cells, _df_recovered_sgRNAs
