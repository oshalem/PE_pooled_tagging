import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import skimage.filters as sf
from skimage.morphology import skeletonize
from skimage.morphology import disk
from scipy import ndimage as sn


def Norm(_img):
    return (_img - _img.min()) / (_img.max() - _img.min())

def Crop_Cell_Window(_I, _J, _cell_mask):

    _cell_number = _cell_mask[_I, _J]
    _single_cell_mask = _cell_mask == _cell_number

    _i_project = np.sum(_single_cell_mask, axis=1) > 0
    _i_min = np.argmax(_i_project)
    _i_max = np.argmax(np.cumsum(_i_project))

    _j_project = np.sum(_single_cell_mask, axis=0) > 0
    _j_min = np.argmax(_j_project)
    _j_max = np.argmax(np.cumsum(_j_project))

    return _i_min, _i_max, _j_min, _j_max

def Smooth_Cell_Outline(_crop):

    K1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
    conved = sn.convolve(_crop, K1)
    mask_edges = np.array((conved == 1) + (conved == 2) + (conved == 3), dtype=float)
    mask_edges = sn.gaussian_filter(mask_edges, sigma=3)
    th = sf.threshold_otsu(mask_edges)
    mask_edges_2 = mask_edges > th
    # mask_edges_2 = sn.convolve(mask_edges_2, K1)
    mask_edges_2 = skeletonize(mask_edges_2)

    return mask_edges_2

def Smooth_Cell(_crop, sigma=10):

    pw = 2 * sigma
    _crop = np.pad(_crop, [(pw, pw), (pw, pw)], mode='constant')

    mask_edges = sn.gaussian_filter(np.array(_crop, dtype=float), sigma=sigma)
    th = sf.threshold_otsu(mask_edges)
    mask_edges_2 = np.array(mask_edges > th, dtype=float)

    mask_edges_2 = mask_edges_2[pw:-pw, pw:-pw]

    return mask_edges_2

def Expand_Mask(_crop, deg=5):

    temp = np.array(_crop, dtype=int)

    expanded = sf.rank.maximum(temp, selem=disk(deg))

    expanded = expanded > 0

    return expanded

def Erode_Cell(_crop, deg=3):

    eroded = sf.rank.minimum(np.array(_crop, dtype=int), selem=disk(deg))

    return eroded

def Get_Last_2D(data):

    if data.ndim <= 2:
        return data
    slc = [0] * (data.ndim - 2)
    slc += [slice(None), slice(None)]

    return data[slc]

def Select_Album_Cells(_df_geno_sgRNA, _n_cell_album, method='random'):

    if len(_df_geno_sgRNA) > _n_cell_album:
        n_itr = _n_cell_album
    else:
        n_itr = len(_df_geno_sgRNA)

    if method == 'brightest':

        _df_geno_sgRNA = _df_geno_sgRNA.sort_values('ints_avg', ascending=False)

        _out = _df_geno_sgRNA.iloc[:n_itr]

    if method == 'random':

        _out = _df_geno_sgRNA.iloc[np.random.randint(len(_df_geno_sgRNA), size=n_itr)]

    return _out

def Blue_Black_CMAP():

    colors = [(0, 0, 0), (0, 0, 1)]  # R -> G -> B
    n_bins = [3, 6, 10, 100]

    for n_bin in n_bins:
        _cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=n_bin)

    return _cmap

def Normalize_Enhance_Constrast(_img):

    flat = np.ravel(_img)
    flat = flat[flat != 0]
    min_ = flat.min()
    max_ = flat.max()
    _img_out = (_img - flat.min()) / (flat.max() - flat.min())

    return(_img_out)

def Make_Cell_Image_From_CSV(I_, J_, _img, _mask, nuc_mask=np.array([None]), padding=0, erode=None, crop_background=True):

    # h, w = Get_Last_2D(_img).shape
    h = _img.shape[-1]
    w = _img.shape[-2]

    i_min_, i_max_, j_min_, j_max_ = Crop_Cell_Window(I_, J_, _mask)

    if i_min_ >= padding and i_max_ < h - padding and j_min_ >= padding and j_max_ < w - padding:

        _crop = _img[:, i_min_ - padding: i_max_ + padding, j_min_ - padding: j_max_ + padding]

        _mask = _mask == _mask[I_, J_]
        _mask_crop = np.array(_mask[i_min_ - padding: i_max_ + padding, j_min_ - padding: j_max_ + padding], dtype=int)

        _mask_smooth = Smooth_Cell(_mask_crop)

        if nuc_mask.any() != None:
            nuc_paddin = padding

            _nuc_mask = nuc_mask == nuc_mask[I_, J_]
            _nuc_mask_crop = np.array(_nuc_mask[i_min_ - nuc_paddin: i_max_ + nuc_paddin, j_min_ - nuc_paddin: j_max_ + nuc_paddin], dtype=int)

            _nuc_mask_crop = Smooth_Cell(_nuc_mask_crop)

        if erode != None:
            _mask_smooth = Erode_Cell(_mask_smooth, deg=erode)

        n_ch = len(_crop)
        h_m, w_m = _mask_smooth.shape
        _mask_array = np.zeros([n_ch, h_m, w_m], dtype=int)
        for m in range(n_ch):
            _mask_array[m] = _mask_smooth

        if crop_background:
            _final_img = _crop * _mask_array
        else:
            _final_img = _crop
        # final_cell_img = Normalize_Enhance_Contrast(final_cell_img)

    else:

        _final_img = np.array([None])
        _mask_smooth = np.array([None])
        _nuc_mask_crop = np.array([None])

    if nuc_mask.any() != None:
        return _final_img, _mask_smooth, _nuc_mask_crop
    else:
        return _final_img, _mask_smooth

def Outline(_I, _J, _mask, padding=0):

    i_min_, i_max_, j_min_, j_max_ = Crop_Cell_Window(_I, _J, _mask)

    h, w = Get_Last_2D(_mask).shape

    if i_min_ >= padding and i_max_ < h - padding and j_min_ >= padding and j_max_ < w - padding:

        _mask = _mask == _mask[_I, _J]
        _single_cell = np.array(_mask[i_min_ - padding: i_max_ + padding, j_min_ - padding: j_max_ + padding], dtype=int)

        outline_cell = Smooth_Cell_Outline(_single_cell)
        pts_cell = np.where(outline_cell == 1)

        _x= pts_cell[1]
        _y = pts_cell[0]

    else:

        _x = []
        _y = []

    return _x, _y

def Outline_From_Mask(_single_cell):

    outline_cell = Smooth_Cell_Outline(_single_cell)
    pts_cell = np.where(outline_cell == 1)

    _x= pts_cell[1]
    _y = pts_cell[0]

    return _x, _y

def Find_Guides_With_Localizations(_df_geno, _df_local, _loc_name):
    sgRNA_list = np.unique(_df_geno['sgRNA'])
    _new_list = np.empty([0])
    for i in range(len(sgRNA_list)):
        loc = _df_local['localization'].iloc[
            np.squeeze(np.where(str(sgRNA_list[i]).split('_')[0] == _df_local['gene'])[0])]
        if isinstance(loc, str):
            if np.sum(loc == np.array([_loc_name])) > 0:
                _new_list = np.append(_new_list, str(sgRNA_list[i]))

    return _new_list

def Save_Cell_Image(_img_cell_final, _cell_name, _save_path):

    h = _img_cell_final.shape[0]
    w = _img_cell_final.shape[1]

    fig = plt.figure(frameon=False)

    ax = fig.subplots(1)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.set_axis_off()
    fig.set_size_inches(w, h)
    fig.add_axes(ax)
    ax.imshow(_img_cell_final)
    fig.savefig(os.path.join(_save_path, _cell_name + '.png'), dpi=1)

    fig.clf()
    plt.close('all')
    matplotlib.use('Agg')

    fig.clf()
    plt.close('all')
    matplotlib.use('Agg')
