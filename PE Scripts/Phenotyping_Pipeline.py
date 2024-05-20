import numpy as np
import pandas as pd
import argparse
import os
import In_Situ_Functions as isf
import Mapping_Functions as mf
from tqdm import tqdm

if __name__ == '__main__':

    def Crop_Cell(_p_40X, _img_40X, _cells):

        _cell_number = _cells[_p_40X[0, 1], _p_40X[0, 2]]
        _single_cell_mask = _cells == _cell_number

        i_project = np.sum(_single_cell_mask, axis=1) > 0
        i_min = np.argmax(i_project)
        i_max = np.argmax(np.cumsum(i_project))

        j_project = np.sum(_single_cell_mask, axis=0) > 0
        j_min = np.argmax(j_project)
        j_max = np.argmax(np.cumsum(j_project))

        _cell_crop = np.array(_img_40X[:, i_min:i_max, j_min:j_max] * _single_cell_mask[i_min:i_max, j_min:j_max], dtype=int)

        return _cell_crop, i_min, i_max, j_min, j_max

    def Get_DOF(_n_well):

        if _n_well == 1:
            _DOF = [1219.91419, -29.5278587,  0.00485282143,  3.95660683, 3.95711362]
        if _n_well == 2:
            _DOF = [1224.47211, -21.5828309,  0.00162262977,  3.95657406, 3.95676618]
        if _n_well == 3:
            _DOF = [1231.41981, -26.7831177,  0.00685868023,  3.95674868, 3.95718144]
        if _n_well == 4:
            _DOF = [1232.27147, -21.2944480,  0.0116983858,  3.95671590, 3.95695640]
        if _n_well == 5:
            _DOF = [1231.88963, -23.2054946,  0.0107022903,  3.95671522, 3.95698056]
        if _n_well == 6:
            _DOF = [1231.47162, -28.5139357,  0.0137829851,  3.95668964, 3.95719910]

        return _DOF

    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=int)
    args = parser.parse_args()

    B = args.batch
    batch_length = 10000

    M_10X = np.load('M_10X.npy')
    M_40X = np.load('M_40X.npy')

    df_name = 'Cells_Genotyped_4.csv'
    df_path = ''
    full_path = os.path.join(df_path, df_name)
    path_40X = ''

    df_geno = pd.read_csv(full_path)
    df_geno = df_geno.iloc[B * batch_length : (B + 1) * batch_length]

    N = len(df_geno)

    tile_40X = -1 * np.ones([N], dtype=int)
    i_nuc_40X = -1 * np.ones([N], dtype=int)
    j_nuc_40X = -1 * np.ones([N], dtype=int)

    area_nuc = -1 * np.ones([N], dtype=int)
    area_cell = -1 * np.ones([N], dtype=int)

    ints_tot_DAPI = -1 * np.ones([N], dtype=int)
    ints_tot_GFP_100 = -1 * np.ones([N], dtype=int)
    ints_tot_GFP_300 = -1 * np.ones([N], dtype=int)
    ints_tot_Syto85 = -1 * np.ones([N], dtype=int)
    ints_tot_Phal = -1 * np.ones([N], dtype=int)

    ints_avg_DAPI = -1 * np.ones([N], dtype=int)
    ints_avg_GFP_100 = -1 * np.ones([N], dtype=int)
    ints_avg_GFP_300 = -1 * np.ones([N], dtype=int)
    ints_avg_Syto85 = -1 * np.ones([N], dtype=int)
    ints_avg_Phal = -1 * np.ones([N], dtype=int)

    ints_std_DAPI = -1 * np.ones([N], dtype=int)
    ints_std_GFP_100 = -1 * np.ones([N], dtype=int)
    ints_std_GFP_300 = -1 * np.ones([N], dtype=int)
    ints_std_Syto85 = -1 * np.ones([N], dtype=int)
    ints_std_Phal = -1 * np.ones([N], dtype=int)

    i_min_list = -1 * np.ones([N], dtype=int)
    i_max_list = -1 * np.ones([N], dtype=int)
    j_min_list = -1 * np.ones([N], dtype=int)
    j_max_list = -1 * np.ones([N], dtype=int)

    for i in tqdm(range(len(df_geno))):

        n_well = int(df_geno['well'].iloc[i])
        T_10X = int(df_geno['tile'].iloc[i])
        I_10X = int(df_geno['i_nuc'].iloc[i])
        J_10X = int(df_geno['j_nuc'].iloc[i])

        DOF = Get_DOF(n_well)

        try:
            P_10X = mf.Local_to_Global(np.array([[T_10X, I_10X, J_10X]]), M_10X, [2304, 2304])
            P_40X = mf.model_TRS(P_10X, DOF, angle='degree')
            p_40X = mf.Global_to_Local(P_40X, M_40X, [2304, 2304])

            p_40X = np.squeeze(p_40X)
            T_40X = p_40X[0]
            I_40X = p_40X[1]
            J_40X = p_40X[2]

            tile_40X[i] = int(T_40X)
            i_nuc_40X[i] = int(I_40X)
            j_nuc_40X[i] = int(J_40X)

            if T_40X >= 0:

                img_40X = isf.InSitu.Import_ND2_by_Tile_and_Well(T_40X, n_well, os.path.join(path_40X, 'phenotyping'))

                cells = np.load(os.path.join(path_40X, 'segmented/40X/cells/well_' + str(n_well), 'Seg_Cells-Well_' + str(n_well) + '_Tile_' + str(T_40X) + '.npy'))
                nucs = np.load(os.path.join(path_40X, 'segmented/40X/nucs/well_' + str(n_well), 'Seg_Nuc-Well_' + str(n_well) + '_Tile_' + str(T_40X) + '.npy'))

                crop_nuc, _, _, _, _ = Crop_Cell(np.array([[T_40X, I_40X, J_40X]]), img_40X, nucs[1])
                crop, i_min, i_max, j_min, j_max = Crop_Cell(np.array([[T_40X, I_40X, J_40X]]), img_40X, cells[1])
                c, h, w = crop.shape

            if h < 500 and w < 500:

                crop_DAPI = crop_nuc[-1]
                crop_GFP_100 = crop[3]
                crop_GFP_300 = crop[2]
                crop_Syto85 = crop[1]
                crop_Phal = crop[0]

                flat_DAPI = np.ravel(crop_DAPI)
                flat_DAPI = flat_DAPI[flat_DAPI != 0]

                flat_GFP_100 = np.ravel(crop_GFP_100)
                flat_GFP_100 = flat_GFP_100[flat_GFP_100 != 0]

                flat_GFP_300 = np.ravel(crop_GFP_300)
                flat_GFP_300 = flat_GFP_300[flat_GFP_300 != 0]

                flat_Syto85 = np.ravel(crop_Syto85)
                flat_Syto85 = flat_Syto85[flat_Syto85 != 0]

                flat_Phal = np.ravel(crop_Phal)
                flat_Phal = flat_Phal[flat_Phal != 0]

                area_nuc[i] = len(flat_DAPI)
                area_cell[i] = len(flat_Phal)

                ints_tot_DAPI[i] = int(np.sum(flat_DAPI))
                ints_tot_GFP_100[i] = int(np.sum(flat_GFP_100))
                ints_tot_GFP_300[i] = int(np.sum(flat_GFP_300))
                ints_tot_Syto85[i] = int(np.sum(flat_Syto85))
                ints_tot_Phal[i] = int(np.sum(flat_Phal))

                ints_avg_DAPI[i] = int(np.mean(flat_DAPI))
                ints_avg_GFP_100[i] = int(np.mean(flat_GFP_100))
                ints_avg_GFP_300[i] = int(np.mean(flat_GFP_300))
                ints_avg_Syto85[i] = int(np.mean(flat_Syto85))
                ints_avg_Phal[i] = int(np.mean(flat_Phal))

                ints_std_DAPI[i] = int(np.std(flat_DAPI))
                ints_std_GFP_100[i] = int(np.std(flat_GFP_100))
                ints_std_GFP_300[i] = int(np.std(flat_GFP_300))
                ints_std_Syto85[i] = int(np.std(flat_Syto85))
                ints_std_Phal[i] = int(np.std(flat_Phal))

                i_min_list[i] = i_min
                i_max_list[i] = i_max
                j_min_list[i] = j_min
                j_max_list[i] = j_max

        except:
           print('Cannot find specific cell. Well:', n_well, 'Tile:', T_10X)

    df_geno['tile_40X'] = tile_40X
    df_geno['i_nuc_40X'] = i_nuc_40X
    df_geno['j_nuc_40X'] = j_nuc_40X

    df_geno['area_nuc'] = area_nuc
    df_geno['area_cell'] = area_cell

    df_geno['ints_tot_DAPI'] = ints_tot_DAPI
    df_geno['ints_tot_GFP_100'] = ints_tot_GFP_100
    df_geno['ints_tot_GFP_300'] = ints_tot_GFP_300
    df_geno['ints_tot_Syto85'] = ints_tot_Syto85
    df_geno['ints_tot_Phal'] = ints_tot_Phal

    df_geno['ints_avg_DAPI'] = ints_avg_DAPI
    df_geno['ints_avg_GFP_100'] = ints_avg_GFP_100
    df_geno['ints_avg_GFP_300'] = ints_avg_GFP_300
    df_geno['ints_avg_Syto85'] = ints_avg_Syto85
    df_geno['ints_avg_Phal'] = ints_avg_Phal

    df_geno['ints_std_DAPI'] = ints_std_DAPI
    df_geno['ints_std_GFP_100'] = ints_std_GFP_100
    df_geno['ints_std_GFP_300'] = ints_std_GFP_300
    df_geno['ints_std_Phal'] = ints_std_Phal
    df_geno['ints_std_Syto85'] = ints_std_Syto85

    df_geno['i_min'] = i_min_list
    df_geno['i_max'] = i_max_list
    df_geno['j_min'] = j_min_list
    df_geno['j_max'] = j_max_list

    file_name = 'Cells_Phenotyped_4'
    df_geno.to_csv('phenotyping_results_4/' + file_name + '_' + str(B) + '.csv', index=False)









