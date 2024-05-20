import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import In_Situ_Functions as isf
import Album_Functions as af
import matplotlib
import argparse
matplotlib.use('Agg')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=int)
    args = parser.parse_args()

    B = args.batch

    # Divide total number of cells into batches of 10000 to parallelize single cell image generation
    batch_length = 10000

    path_name = ''
    path_df = 'Cells_Phenotyped.csv'
    save_path = 'single_cells'
    dataframe_save = 'single_cells_dataframes'

    # path_df = 'Cells_Phenotyped_4_Lenti_reads-et-matched_reads-gtoet-2_avgints300-gtoet-1000.csv'
    # save_path = 'single_cells_4_Lenti'
    # dataframe_save = 'single_cells_dataframes_4_Lenti'

    seg_path = path_name + 'segmented/40X'
    path_40X = path_name + 'phenotyping'

    df = pd.read_csv(path_df)
    df_sub = df.iloc[B * batch_length: (B + 1) * batch_length]

    image_list = -1 * np.ones([len(df_sub)], dtype=object)
    for i in tqdm(range(len(df_sub))):

        n_well = int(df_sub['well'].iloc[i])
        T_40X = int(df_sub['tile_40X'].iloc[i])
        I_40X = int(df_sub['i_nuc_40X'].iloc[i])
        J_40X = int(df_sub['j_nuc_40X'].iloc[i])
        sgRNA = df_sub['sgRNA'].iloc[i]
        Cell = int(df_sub['cell'].iloc[i])
        T_10X = int(df_sub['tile'].iloc[i])

        # try:

        cell_mask_path = seg_path + '/cells/well_' + str(n_well)
        nuc_mask_path = seg_path + '/nucs/well_' + str(n_well)
        nucs_mask = np.load(os.path.join(nuc_mask_path, 'Seg_Nuc-Well_' + str(n_well) + '_Tile_' + str(T_40X) + '.npy'))
        cells_mask = np.load(os.path.join(cell_mask_path, 'Seg_Cells-Well_' + str(n_well) + '_Tile_' + str(T_40X) + '.npy'))

        img_40X = isf.InSitu.Import_ND2_by_Tile_and_Well(T_40X, n_well, path_40X)

        img_cell, cells_mask_crop, nucs_mask_crop = af.Make_Cell_Image_From_CSV(I_40X, J_40X, img_40X, cells_mask[1], nuc_mask=nucs_mask[1], padding=10, crop_background=False)

        if img_cell.any() != None:

            img_cell = (img_cell - img_cell.min()) / (img_cell.max() - img_cell.min())

            cell_name = sgRNA + '_W-' + str(n_well) + '_T-' + str(T_10X) + '_C-' + str(Cell)
            image_list[i] = cell_name + '.png'

            _, h, w = img_cell.shape

            img_cell_final = np.zeros([h, w, 3])
            img_cell_final[:, :, 0] = af.Norm(img_cell[3])
            img_cell_final[:, :, 1] = 0.1 * af.Norm(nucs_mask_crop)
            img_cell_final[:, :, 2] = 0.1 * af.Norm(cells_mask_crop)

            af.Save_Cell_Image(img_cell_final, cell_name, save_path)

        # except:

            # print(cell_name, 'Cannot be found')


    df_sub['image'] = image_list
    df_sub.to_csv(os.path.join(dataframe_save, 'single_cell_dataframes_' + str(B) + '.csv'), index=False)