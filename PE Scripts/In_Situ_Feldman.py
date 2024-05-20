import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import skimage.io
import skimage.morphology
from scipy import ndimage as ndi
import decorator
import sys
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
warnings.filterwarnings('ignore', message='regionprops and image moments')
warnings.filterwarnings('ignore', message='non-tuple sequence for multi')
warnings.filterwarnings('ignore', message='precision loss when converting')


# All functions here were taken from Feldman et al: 'Optical Pooled Screens in Human Cells', Cell 2019
# Link: https://www.sciencedirect.com/science/article/pii/S0092867419310670
# Github: https://github.com/feldman4/OpticalPooledScreens/tree/master


@decorator.decorator
def applyIJ(f, arr, *args, **kwargs):
    """Apply a function that expects 2D input to the trailing two
    dimensions of an array. The function must output an array whose shape
    depends only on the input shape.
    """
    h, w = arr.shape[-2:]
    reshaped = arr.reshape((-1, h, w))

    # kwargs are not actually getting passed in?
    arr_ = [f(frame, *args, **kwargs) for frame in reshaped]

    output_shape = arr.shape[:-2] + arr_[0].shape
    return np.array(arr_).reshape(output_shape)


class Snake:
    """Container class for methods that act directly on data (names start with
    underscore) and methods that act on arguments from snakemake (e.g., filenames
    provided instead of image and table data). The snakemake methods (no underscore)
    are automatically loaded by `Snake.load_methods`.
    """

    # ALIGNMENT AND SEGMENTATION

    @staticmethod
    def remove_channels(data, remove_index):
        """Remove channel or list of channels from array of shape (..., CHANNELS, I, J).
        """
        channels_mask = np.ones(data.shape[-3], dtype=bool)
        channels_mask[remove_index] = False
        data = data[..., channels_mask, :, :]
        return data

    @staticmethod
    def _align_SBS(data, method='DAPI', upsample_factor=2, window=2, cutoff=1, align_channels=slice(1, None), keep_trailing=False):
        """Rigid alignment of sequencing cycles and channels.

        Parameters
        ----------

        data : numpy array
            Image data, expected dimensions of (CYCLE, CHANNEL, I, J).

        method : {'DAPI','SBS_mean'}, default 'DAPI'
            Method for aligning 'data' across cycles. 'DAPI' uses cross-correlation between subsequent cycles
            of DAPI images, assumes sequencing channels are aligned to DAPI images. 'SBS_mean' uses the
            mean background signal from the SBS channels to determine image offsets between cycles of imaging,
            again using cross-correlation.

        upsample_factor : int, default 2
            Subpixel alignment is done if `upsample_factor` is greater than one (can be slow).
            Parameter passed to skimage.feature.register_translation.

        window : int, default 2
            A centered subset of data is used if `window` is greater than one. The size of the removed border is
            int((x/2.) * (1 - 1/float(window))).

        cutoff : float, default 1
            Threshold for removing extreme values from SBS channels when using method='SBS_mean'. Channels are normalized
            to the 70th percentile, and normalized values greater than `cutoff` are replaced by `cutoff`.

        align_channels : slice object or None, default slice(1,None)
            If not None, aligns channels (defined by the passed slice object) to each other within each cycle. If
            None, does not align channels within cycles. Useful in particular for cases where images for all stage
            positions are acquired for one SBS channel at a time, i.e., acquisition order of channels(positions).

        keep_trailing : boolean, default True
            If True, keeps only the minimum number of trailing channels across cycles. E.g., if one cycle contains 6 channels,
            but all others have 5, only uses trailing 5 channels for alignment.

        n : int, default 1
            The first SBS channel in `data`.

        Returns
        -------

        aligned : numpy array
            Aligned image data, same dimensions as `data` unless `data` contained different numbers of channels between cycles
            and keep_trailing=True.
        """
        data = np.array(data)
        if keep_trailing:
            valid_channels = min([len(x) for x in data])
            data = np.array([x[-valid_channels:] for x in data])

        assert data.ndim == 4, 'Input data must have dimensions CYCLE, CHANNEL, I, J'

        # align SBS channels for each cycle
        aligned = data.copy()

        if align_channels is not None:
            align_it = lambda x: Align.align_within_cycle(
                x, window=window, upsample_factor=upsample_factor)
            aligned[:, align_channels] = np.array(
                [align_it(x) for x in aligned[:, align_channels]])

        if method == 'DAPI':
            # align cycles using the DAPI channel
            aligned = Align.align_between_cycles(aligned, channel_index=-1,
                                                 window=window, upsample_factor=upsample_factor)

        return aligned

    @staticmethod
    def _align_by_DAPI(data_1, data_2, channel_index=0, upsample_factor=2, autoscale=True):
        """Align the second image to the first, using the channel at position
        `channel_index`. The first channel is usually DAPI.

        Parameters
        ----------

        data_1 : numpy array
            Image data to align to, expected dimensions of (CHANNEL, I, J).

        data_2 : numpy array
            Image data to align, expected dimensions of (CHANNEL, I, J).

        channel_index : int, default 0
            DAPI channel index

        upsample_factor : int, default 2
            Subpixel alignment is done if `upsample_factor` is greater than one (can be slow).
            Parameter passed to skimage.feature.register_translation.

        autoscale : bool, default True
            Automatically scale `data_2` prior to alignment. Offsets are applied to
            the unscaled image so no resolution is lost.

        Returns
        -------

        aligned : numpy array
            `data_2` with calculated offsets applied to all channels.
        """
        images = [data_1[channel_index], data_2[channel_index]]
        if autoscale:
            images[1] = Utils.match_size(images[1], images[0])

        _, offset = Align.calculate_offsets(images, upsample_factor=upsample_factor)
        if autoscale:
            offset *= data_2.shape[-1] / data_1.shape[-1]

        offsets = [offset] * len(data_2)
        aligned = Align.apply_offsets(data_2, offsets)
        return aligned

    @staticmethod
    def _transform_log(data, sigma=1, skip_index=None):
        """Apply Laplacian-of-Gaussian filter from scipy.ndimage.

        Parameters
        ----------

        data : numpy array
            Aligned SBS image data, expected dimensions of (CYCLE, CHANNEL, I, J).

        sigma : float, default 1
            size of gaussian kernel used in Laplacian-of-Gaussian filter

        skip_index : None or int, default None
            If an int, skips transforming a channel (e.g., DAPI with `skip_index=0`).

        Returns
        -------

        loged : numpy array
            LoG-ed `data`
        """
        data = np.array(data)
        loged = log_ndi(data, sigma=sigma)
        if skip_index is not None:
            loged[..., skip_index, :, :] = data[..., skip_index, :, :]
        return loged

    @staticmethod
    def _compute_std(data, remove_index=None):
        """Use standard deviation over cycles, followed by mean across channels
        to estimate sequencing read locations. If only 1 cycle is present, takes
        standard deviation across channels.

        Parameters
        ----------

        data : numpy array
            LoG-ed SBS image data, expected dimensions of (CYCLE, CHANNEL, I, J).

        remove_index : None or int, default None
            Index of `data` to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI)

        Returns
        -------

        consensus : numpy array
            Standard deviation score for each pixel, dimensions of (I,J).
        """
        if remove_index is not None:
            data = Snake.remove_channels(data, remove_index)

        # for 1-cycle experiments
        if len(data.shape) == 3:
            data = data[:, None, ...]

        # leading_dims = tuple(range(0, data.ndim - 2))
        # consensus = np.std(data, axis=leading_dims)
        consensus = np.std(data, axis=0).mean(axis=0)

        return consensus

    @staticmethod
    def _find_peaks(data, width=5, remove_index=None):
        """Find local maxima and label by difference to next-highest neighboring
        pixel. Conventionally this is used to estimate SBS read locations by inputting
        the standard deviation score as returned by Snake.compute_std().

        Parameters
        ----------

        data : numpy array
            2D image data

        width : int, default 5
            Neighborhood size for finding local maxima.

        remove_index : None or int, default None
            Index of `data` to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI)

        Returns
        -------

        peaks : numpy array
            Local maxima scores, dimensions same as `data`. At a maximum, the value is max - min in the defined
            neighborhood, elsewhere zero.
        """
        if remove_index is not None:
            data = Snake.remove_channels(data, remove_index)

        if data.ndim == 2:
            data = [data]

        peaks = [find_peaks(x, n=width)
                 if x.max() > 0 else x
                 for x in data]
        peaks = np.array(peaks).squeeze()
        return peaks

    @staticmethod
    def _max_filter(data, width, remove_index=None):
        """Apply a maximum filter in a window of `width`. Conventionally operates on Laplacian-of-Gaussian
        filtered SBS data, dilating sequencing channels to compensate for single-pixel alignment error.

        Parameters
        ----------

        data : numpy array
            Image data, expected dimensions of (..., I, J) with up to 4 total dimenions.

        width : int
            Neighborhood size for max filtering

        remove_index : None or int, default None
            Index of `data` to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI)

        Returns
        -------

        maxed : numpy array
            Maxed `data` with preserved dimensions.
        """
        import scipy.ndimage.filters

        if data.ndim == 2:
            data = data[None, None]
        if data.ndim == 3:
            data = data[None]

        if remove_index is not None:
            data = Snake.remove_channels(data, remove_index)

        maxed = scipy.ndimage.filters.maximum_filter(data, size=(1, 1, width, width))

        return maxed


class Utils:

    @staticmethod
    def match_size(image, target, order=None):
        """Resize image to target without changing data range or type.
        """
        from skimage.transform import resize
        return (resize(image, target.shape, preserve_range=True, order=order)
                .astype(image.dtype))


class Align:
    """Alignment redux, used by snakemake.
    """
    @staticmethod
    def normalize_by_percentile(data_, q_norm=70):
        shape = data_.shape
        shape = shape[:-2] + (-1,)
        p = np.percentile(data_, q_norm, axis=(-2, -1))[..., None, None]
        normed = data_ / p
        return normed

    @staticmethod
    @applyIJ
    def filter_percentiles(data, q1, q2):
        """Replaces data outside of percentile range [q1, q2]
        with uniform noise over the range [q1, q2]. Useful for
        eliminating alignment artifacts due to bright features or
        regions of zeros.
        """
        x1, x2 = np.percentile(data, [q1, q2])
        mask = (x1 > data) | (x2 < data)
        return Align.fill_noise(data, mask, x1, x2)

    @staticmethod
    @applyIJ
    def filter_values(data, x1, x2):
        """Replaces data outside of value range [x1, x2]
        with uniform noise over the range [x1, x2]. Useful for
        eliminating alignment artifacts due to bright features or
        regions of zeros.
        """
        mask = (x1 > data) | (x2 < data)
        return Align.fill_noise(data, mask, x1, x2)

    @staticmethod
    def fill_noise(data, mask, x1, x2):
        filtered = data.copy()
        rs = np.random.RandomState(0)
        filtered[mask] = rs.uniform(x1, x2, mask.sum()).astype(data.dtype)
        return filtered

    @staticmethod
    def calculate_offsets(data_, upsample_factor, target_index=0):
        target = data_[target_index]
        offsets = []
        for i, src in enumerate(data_):
            # if i == 0:
            #     offsets += [(0, 0)]
            # else:
            offset, _, _ = skimage.registration.phase_cross_correlation(
                            src, target, upsample_factor=upsample_factor)
            offsets += [offset]
        return np.array(offsets)

    @staticmethod
    def apply_offsets(data_, offsets):
        warped = []
        for frame, offset in zip(data_, offsets):
            if offset[0] == 0 and offset[1] == 0:
                warped += [frame]
            else:
                # skimage has a weird (i,j) <=> (x,y) convention
                st = skimage.transform.SimilarityTransform(translation=offset[::-1])
                frame_ = skimage.transform.warp(frame, st, preserve_range=True)
                warped += [frame_.astype(data_.dtype)]

        return np.array(warped)

    @staticmethod
    def align_within_cycle(data_, upsample_factor=4, window=1, q1=0, q2=90):
        filtered = Align.filter_percentiles(Align.apply_window(data_, window), q1=q1, q2=q2)

        offsets = Align.calculate_offsets(filtered, upsample_factor=upsample_factor)

        return Align.apply_offsets(data_, offsets)

    @staticmethod
    def align_between_cycles(data, channel_index, upsample_factor=4, window=1, return_offsets=False):
        # offsets from target channel
        target = Align.apply_window(data[:, channel_index], window)
        offsets = Align.calculate_offsets(target, upsample_factor=upsample_factor, target_index=-1)

        # apply to all channels
        warped = []
        for data_ in data.transpose([1, 0, 2, 3]):
            warped += [Align.apply_offsets(data_, offsets)]

        aligned = np.array(warped).transpose([1, 0, 2, 3])
        if return_offsets:
        	return aligned, offsets
        else:
        	return aligned

    @staticmethod
    def apply_window(data, window):
        height, width = data.shape[-2:]
        find_border = lambda x: int((x/2.) * (1 - 1/float(window)))
        i, j = find_border(height), find_border(width)
        return data[..., i:height - i, j:width - j]


class QC:

    @staticmethod
    def plot_mapping_vs_threshold(df_reads, barcodes, threshold_var='peak', ax=None):
        """Plot the mapping rate and number of mapped spots against varying thresholds of
        peak intensity, quality score, or a user-defined metric.

        Parameters
        ----------
        df_reads : pandas DataFrame
            Table of extracted reads from Snake.call_reads(). Can be concatenated results from
            multiple tiles, wells, etc.

        barcodes : list or set of strings
            Expected barcodes from the pool library design.

        threshold_var : string, default 'peak'
            Variable to apply varying thresholds to for comparing mapping rates. Standard variables are
            'peak' and 'QC_min'. Can also use a user-defined variable, but must be a column of the df_reads
            table.

        ax : None or matplotlib axis object, default None
            Optional. If not None, this is an axis object to plot on. Helpful when plotting on
            a subplot of a larger figure.

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments passed to sns.lineplot()

        Returns
        -------
        df_summary : pandas DataFrame
            Summary table of thresholds and associated mapping rates, number of spots mapped used for plotting.
        """
        # exclude spots not in cells
        df_passed = df_reads.copy().query('cell>0')

        # map reads
        df_passed.loc[:, 'mapped'] = df_passed['barcode'].isin(barcodes)

        # define thresholds range
        if df_reads[threshold_var].max() < 100:
            thresholds = np.array(range(0, int(np.quantile(df_passed[threshold_var], q=0.99) * 1000))) / 1000
        else:
            thresholds = list(range(0, int(np.quantile(df_passed[threshold_var], q=0.99)), 10))

        # iterate over thresholds
        mapping_rate = []
        spots_mapped = []
        for threshold in thresholds:
            df_passed = df_passed.query('{} > @threshold'.format(threshold_var))
            spots_mapped.append(df_passed[df_passed['mapped']].pipe(len))
            mapping_rate.append(df_passed[df_passed['mapped']].pipe(len) / df_passed.pipe(len))

        df_summary = pd.DataFrame(np.array([thresholds, mapping_rate, spots_mapped]).T,
                                  columns=['{}_threshold'.format(threshold_var), 'mapping_rate', 'mapped_spots'])

        # plot
        if not ax:
            ax = sns.lineplot(data=df_summary, x='{}_threshold'.format(threshold_var), y='mapping_rate')
        else:
            sns.lineplot(data=df_summary, x='{}_threshold'.format(threshold_var), y='mapping_rate', ax=ax)

        ax.set_ylabel('mapping rate', fontsize=18)
        ax.set_xlabel('{} threshold'.format(threshold_var), fontsize=18)
        ax_right = ax.twinx()
        sns.lineplot(data=df_summary, x='{}_threshold'.format(threshold_var), y='mapped_spots', ax=ax_right, color='coral')
        ax_right.set_ylabel('mapped spots', fontsize=18)
        plt.legend(ax.get_lines() + ax_right.get_lines(), ['mapping rate', 'mapped spots'], loc=7)

        return df_summary

    @staticmethod
    def plot_count_heatmap(df, tile='tile', shape='square', plate='6W', return_summary=False, **kwargs):
        """Plot the count of items in df by well and tile in a convenient plate layout.
        Useful for evaluating cell and read counts across wells. The colorbar label can
        be modified with:
            axes[0,0].get_figure().axes[-1].set_ylabel(LABEL)

        Parameters
        ----------
        df : pandas DataFrame

        tile : str, default 'tile'
            The column name to be used to group tiles, as sometimes 'site' is used.

        shape : str, default 'square'
            Shape of subplot for each well used in plot_plate_heatmap

        plate : {'6W','24W','96W'}
            Plate type for plot_plate_heatmap

        return_summary : boolean, default False
            If true, returns df_summary

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments passed to plot_plate_heatmap()

        Returns
        -------
        df_summary : pandas DataFrame
            DataFrame used for plotting
            optional output, only returns if return_summary=True

        axes : np.array of matplotlib Axes objects
        """
        df_summary = (df.groupby(['well', tile]).size().rename('count').to_frame().reset_index())

        axes = QC.plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)

        if return_summary:
            return df_summary, axes
        return axes

    @staticmethod
    def plot_cell_mapping_heatmap(df_cells, df_sbs_info, barcodes, mapping_to='one', shape='square', plate='6W', return_summary=False, **kwargs):
        """Plot the mapping rate of cells by well and tile in a convenient plate layout.

        Parameters
        ----------
        df_cells : pandas DataFrame
            DataFrame of all cells output from sbs mapping pipeline, e.g., concatenated outputs for all tiles and wells
            of Snake.call_cells().

        df_sbs_info : pandas DataFrame
            DataFrame of all cells segmented from sbs images, e.g., concatenated outputs for all tiles and wells of
            Snake.extract_phenotype_minimal(data_phenotype=nulcei,nuclei=nuclei) often used as sbs_cell_info rule in
            Snakemake.

        barcodes : list or set of strings
            Expected barcodes from the pool library design.

        mapping_to : {'one', 'any'}
            Cells to include as 'mapped'. 'one' only includes cells mapping to a single barcode, 'any' includes cells
            mapping to at least 1 barcode.

        shape : str, default 'square'
            Shape of subplot for each well used in plot_plate_heatmap

        plate : {'6W','24W','96W'}
            Plate type for plot_plate_heatmap

        return_summary : boolean, default False
            If true, returns df_summary

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments passed to plot_plate_heatmap()

        Returns
        -------
        df_summary : pandas DataFrame
            DataFrame used for plotting
            optional output, only returns if return_summary=True

        axes : np.array of matplotlib Axes objects
        """
        df_cells.loc[:, ['mapped_0', 'mapped_1']] = df_cells[['cell_barcode_0', 'cell_barcode_1']].isin(barcodes).values

        df = (df_sbs_info[['well', 'tile', 'cell']].merge(df_cells[['well', 'tile', 'cell', 'mapped_0', 'mapped_1']], how='left', on=['well', 'tile', 'cell']))

        if mapping_to == 'one':
            metric = 'fraction of cells mapping to 1 barcode'
            df = df.assign(mapped=lambda x: x[['mapped_0', 'mapped_1']].sum(axis=1) == 1)
        elif mapping_to == 'any':
            metric = 'fraction of cells mapping to >=1 barcode'
            df = df.assign(mapped=lambda x: x[['mapped_0', 'mapped_1']].sum(axis=1) > 0)
        else:
            raise ValueError(f'mapping_to={mapping_to} not implemented')

        df_summary = (df.groupby(['well', 'tile'])['mapped'].value_counts(normalize=True).rename(metric).to_frame().reset_index().query('mapped').drop(columns='mapped'))

        axes = QC.plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)

        if return_summary:
            return df_summary, axes
        return axes

    @staticmethod
    def plot_read_mapping_heatmap(df_reads, barcodes, shape='square', plate='6W', return_summary=False, **kwargs):
        """Plot the mapping rate of reads by well and tile in a convenient plate layout.

        Parameters
        ----------
        df_reads: pandas DataFrame
            DataFrame of all reads output from sbs mapping pipeline, e.g., concatenated outputs for all tiles and wells
            of Snake.call_reads().

        barcodes : list or set of strings
            Expected barcodes from the pool library design.

        shape : str, default 'square'
            Shape of subplot for each well used in plot_plate_heatmap

        plate : {'6W','24W','96W'}
            Plate type for plot_plate_heatmap

        return_summary : boolean, default False
            If true, returns df_summary

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments passed to plot_plate_heatmap()

        Returns
        -------
        df_summary : pandas DataFrame
            DataFrame used for plotting
            optional output, only returns if return_summary=True

        axes : np.array of matplotlib Axes objects
        """

        df_reads.loc[:, 'mapped'] = df_reads['barcode'].isin(barcodes)

        df_summary = (df_reads.groupby(['well', 'tile'])['mapped'].value_counts(normalize=True).rename('fraction of reads mapping').to_frame().reset_index().query('mapped').drop(columns='mapped'))

        axes = QC.plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)

        if return_summary:
            return df_summary, axes
        return axes

    @staticmethod
    def plot_sbs_ph_matching_heatmap(df_merge, df_info, target='sbs', shape='square', plate='6W', return_summary=False, **kwargs):
        """Plot the rate of matching segmented cells between phenotype and SBS datasets by well and tile
        in a convenient plate layout.

        Parameters
        ----------
        df_merge: pandas DataFrame
            DataFrame of all matched cells, e.g., concatenated outputs for all tiles and wells
            of Snake.merge_triangle_hash(). Expects 'tile' and 'cell_0' columns to correspond to phenotype data and
            'site', 'cell_1' columns to correspond to sbs data.

        df_info : pandas DataFrame
            DataFrame of all cells segmented from either phenotype or sbs images, e.g., concatenated outputs for all tiles and wells of
            Snake.extract_phenotype_minimal(data_phenotype=nulcei,nuclei=nuclei) often used as sbs_cell_info rule in
            Snakemake.

        target : {'sbs','phenotype'}
            Which dataset to use as the target, e.g., if target='sbs' plots fraction of cells in each sbs tile that match to
            a phenotype cell. Should match the information stored in df_info; if df_info is a table of all segmented cells from
            sbs tiles then target should be set as 'sbs'.

        shape : str, default 'square'
            Shape of subplot for each well used in plot_plate_heatmap. Default infers shape based on value of target.

        plate : {'6W','24W','96W'}
            Plate type for plot_plate_heatmap

        return_summary : boolean, default False
            If true, returns df_summary

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments passed to plot_plate_heatmap()

        Returns
        -------
        df_summary : pandas DataFrame
            DataFrame used for plotting
            optional output, only returns if return_summary=True

        axes : np.array of matplotlib Axes objects
        """
        if target == 'sbs':
            merge_cols = ['site', 'cell_1']
            source = 'phenotype'
            if not shape:
                shape = '6W_sbs'
        elif target == 'phenotype':
            merge_cols = ['tile', 'cell_0']
            source = 'sbs'
            if not shape:
                shape = '6W_ph'
        else:
            raise ValueError('target = {} not implemented'.format(target))

        df_summary = (df_info
                      .rename(columns={'tile': merge_cols[0], 'cell': merge_cols[1]})
                      [['well'] + merge_cols]
                      .merge(df_merge[['well'] + merge_cols + ['distance']],
                             how='left', on=['well'] + merge_cols)
                      .assign(matched=lambda x: x['distance'].notna())
                      .groupby(['well'] + merge_cols[:1])
                      ['matched']
                      .value_counts(normalize=True)
                      .rename('fraction of {} cells matched to {} cells'.format(target, source))
                      .to_frame()
                      .reset_index()
                      .query('matched==True')
                      .drop(columns='matched')
                      .rename(columns={merge_cols[0]: 'tile'})
                      )
        axes = QC.plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)

        if return_summary:
            return df_summary, axes
        return axes

    @staticmethod
    def plot_plate_heatmap(df, metric=None, shape='square', plate='6W', snake_sites=True, **kwargs):
        """Plot the rate of matching segmented cells between phenotype and SBS datasets by well and tile
        in a convenient plate layout.

        Parameters
        ----------
        df: pandas DataFrame
            Summary DataFrame of values to plot, expects one row for each (well, tile) combination.

        metric : str, default None
            Column of `df` to use for plotting the heatmap. If None, attempts to infer based on column names.

        shape : {'square','6W_ph','6W_sbs',list}, default 'square'
            Shape of subplot for each well.
                'square' infers dimensions of the smallest square that fits the number of sites.
                '6W_ph' and '6W_sbs' use a common  6 well tile map from a Nikon Ti2/Elements set-up with 20X and 10X objectives,
                    respectively.
                Alternatively, a list can be passed containing the number of sites in each row of a tile layout. This is mapped
                    into a centered shape within a rectangle. Unused corners of this rectangle are plotted as NaN. The summation
                    of this list should equal the total number of sites.

        plate : {'6W','24W','96W'}
            Plate type for plot_plate_heatmap

        snake_sites : boolean, default True
            If true, plots tiles in a snake order similar to the order of sites acquired by many high throughput
            microscope systems.

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments passed to matplotlib.pyplot.imshow()

        Returns
        -------
        axes : np.array of matplotlib Axes objects
        """
        import string

        tiles = max(len(df['tile'].unique()), df['tile'].max())

        # define grid for plotting
        if shape == 'square':
            r = c = int(np.ceil(np.sqrt(tiles)))
            grid = np.empty(r * c)
            grid[:] = np.NaN
            grid[:tiles] = range(tiles)
            grid = grid.reshape(r, c)
        else:
            if shape == '6W_ph':
                rows = [7, 13, 17, 21, 25, 27, 29, 31, 33, 33, 35, 35, 37, 37, 39, 39, 39, 41, 41, 41, 41,
                        41, 41, 41, 39, 39, 39, 37, 37, 35, 35, 33, 33, 31, 29, 27, 25, 21, 17, 13, 7]
            elif shape == '6W_sbs':
                rows = [5, 9, 13, 15, 17, 17, 19, 19, 21, 21, 21, 21, 21, 19, 19, 17, 17, 15, 13, 9, 5]
            elif isinstance(shape, list):
                rows = shape
            else:
                raise ValueError('{} shape not implemented, can pass custom shape as a'
                                 'list specifying number of sites per row'.format(shape))

            r, c = len(rows), max(rows)
            grid = np.empty((r, c))
            grid[:] = np.NaN

            next_site = 0
            for row, row_sites in enumerate(rows):
                start = int((c - row_sites) / 2)
                grid[row, start:start + row_sites] = range(next_site, next_site + row_sites)
                next_site += row_sites

        if snake_sites:
            grid[1::2] = grid[1::2, ::-1]

        # infer metric to plot if necessary
        if not metric:
            metric = [col for col in df.columns if col not in ['plate', 'well', 'tile']]
            if len(metric) != 1:
                raise ValueError('Cannot infer metric to plot, can pass metric column name explicitly to metric kwarg')
            metric = metric[0]

        # define subplots layout
        if df['well'].nunique() == 1:
            wells = df['well'].unique()
            fig, axes = plt.subplots(1, 1, figsize=(10, 10))
            axes = np.array([axes])
        elif plate == '6W':
            wells = [f'{r}{c}' for r in string.ascii_uppercase[:2] for c in range(1, 4)]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        elif plate == '24W':
            wells = [f'{r}{c}' for r in string.ascii_uppercase[:4] for c in range(1, 7)]
            fig, axes = plt.subplots(4, 6, figsize=(15, 10))
        elif plate == '96W':
            wells = [f'{r}{c}' for r in string.ascii_uppercase[:8] for c in range(1, 13)]
            fig, axes = plt.subplots(8, 12, figsize=(15, 10))
        else:
            wells = sorted(df['well'].unique())
            nr = nc = int(np.ceil(np.sqrt(len(wells))))
            if (nr - 1) * nc >= len(wells):
                nr -= 1
            fig, axes = plt.subplots(nr, nc, figsize=(15, 15))

        # define colorbar min and max
        cmin, cmax = (df[metric].min(), df[metric].max())

        # plot wells
        for ax, well in zip(axes.reshape(-1), wells):
            values = grid.copy()
            df_well = df.query('well==@well')
            if df_well.pipe(len) > 0:
                for tile in range(tiles):
                    try:
                        values[grid == tile] = df_well.loc[df_well.tile == tile, metric].values[0]
                    except:
                        values[grid == tile] = np.nan
                plot = ax.imshow(values, vmin=cmin, vmax=cmax, **kwargs)
            ax.set_title('Well {}'.format(well), fontsize=24)
            ax.axis('off')

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
        try:
            cbar = fig.colorbar(plot, cax=cbar_ax)
        except:
            # plot variable empty, no data plotted
            raise ValueError('No data to plot')
        cbar.set_label(metric, fontsize=18)
        cbar_ax.yaxis.set_ticks_position('left')

        return axes, cbar


def find_peaks(data, n=5):
    """Finds local maxima. At a maximum, the value is max - min in a
    neighborhood of width `n`. Elsewhere it is zero.
    """
    filters = ndi.filters
    neighborhood_size = (1,) * (data.ndim - 2) + (n, n)
    data_max = filters.maximum_filter(data, neighborhood_size)
    data_min = filters.minimum_filter(data, neighborhood_size)
    peaks = data_max - data_min
    peaks[data != data_max] = 0

    # remove peaks close to edge
    mask = np.ones(peaks.shape, dtype=bool)
    mask[..., n:-n, n:-n] = False
    peaks[mask] = 0

    return peaks

@applyIJ
def log_ndi(data, sigma=1):
    """Apply laplacian of gaussian to each image in a stack of shape
    (..., I, J).
    Extra arguments are passed to scipy.ndimage.filters.gaussian_laplace.
    Inverts output and converts back to uint16.
    """
    f = ndi.filters.gaussian_laplace
    arr_ = -1 * f(data.astype(float), sigma)
    arr_ = np.clip(arr_, 0, 65535) / 65535

    # skimage precision warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return skimage.img_as_uint(arr_)


