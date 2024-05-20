import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, sys
import In_Situ_Functions as isf
import Album_Functions as af
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('batch', type=int)
args = parser.parse_args()
B = args.batch


np.random.seed(seed=2023)
n_cell_album = 50

# df_lib = pd.read_csv('RBP_F2_Bulk_Optimized_Final.csv')
# df = pd.read_csv('Cells_Phenotyped.csv')
# df = pd.read_csv('Cells_Cytoself.csv')
# df = pd.read_csv('Cells_Q-Phenotyped_3.csv')
# df = df_full[df_full['mismatch'] <= 7]
# df = df[df['correct'] >= 3]

# df = df[(df['well'] == 3)]
# df = df[(df['well'] == 3) | (df['well'] == 5) | (df['well'] == 6)]
# df = df[df['reads'] == df['correct']]
# df = df[df['reads'] >= 3]
# df = df[df['matched'] >= 3]
# df = df[df['mismatch'] <= 4]
# df_geno = df[df['ints_avg_GFP_100'] >= 1000]

df = pd.read_csv('Cells_Genotyped.csv')

df_geno = df.copy()

path_40X = 'phenotyping'

df_counts = df_geno['sgRNA'].value_counts()
sgRNA_list = np.sort(df_counts[df_counts >= 0].index)

# batch_length = 20
# sgRNA_list_batch = sgRNA_list[B * batch_length: (B + 1) * batch_length]

for g, sgRNA in tqdm(enumerate(sgRNA_list)):

    df_geno_sgRNA = df_geno[df_geno['sgRNA'] == sgRNA]

    df_geno_sgRNA = df_geno_sgRNA.sample(frac=1, random_state=2023)

    plt.figure(figsize=(24, 12))
    i = 0; j = 0
    while j < n_cell_album and i < len(df_geno_sgRNA):

        n_well = int(df_geno_sgRNA['well'].iloc[i])
        T_40X = int(df_geno_sgRNA['tile_40X'].iloc[i])
        I_40X = int(df_geno_sgRNA['i_nuc_40X'].iloc[i])
        J_40X = int(df_geno_sgRNA['j_nuc_40X'].iloc[i])
        Ints = int(df_geno_sgRNA['ints_avg_GFP_100'].iloc[i])

        if T_40X >= 0:
            # cell_mask_path = 'TLopt_10March2022_Old/segmented/40X/cells/well_' + str(n_well)
            cell_mask_path = 'segmented/40X/cells/well_' + str(n_well)
            cells_mask = np.load(os.path.join(cell_mask_path, 'Seg_Cells-Well_' + str(n_well) + '_Tile_' + str(T_40X) + '.npy'))[1]
            img_40X = isf.InSitu.Import_ND2_by_Tile_and_Well(T_40X, n_well, path_40X)

            img_cell_final = np.array([[None]])

            if cells_mask[I_40X, J_40X] > 0:
                # print('In')
                img_cell_final, _ = af.Make_Cell_Image_From_CSV(I_40X, J_40X, img_40X, cells_mask, padding=4)

            # print('img_cell_final')
            # print(type(img_cell_final))
            # print(img_cell_final.shape)

            if img_cell_final.any() != None:
                plt.subplot(5, 10, j + 1)
                plt.title('W-' + str(int(df_geno_sgRNA['well'].iloc[i])) + '_T-' + str(int(df_geno_sgRNA['tile'].iloc[i])) + '_C-' + str(int(df_geno_sgRNA['cell'].iloc[i])) + '_I-' + str(int(Ints)), fontsize=6)
                # plt.scatter(x_nuc, y_nuc, c='c', s=0.5)
                plt.imshow(img_cell_final[3], cmap='Greys_r')
                # plt.scatter(X, Y, c='r', s=0.001)
                plt.axis('off')
                j = j + 1

        i = i + 1

    # album_name = 'Well_' + str(n_well) + ' - ' + str(local) + ' - ' + str(sgRNA)
    # plt.suptitle(str(local) + '_' + str(sgRNA))
    plt.suptitle(sgRNA)
    # well_' + str(int(n_well)) + '/' +
    plt.savefig(os.path.join('albums_Cropseq_Cytoself', sgRNA + '.png')) # + '_' + str(int(n_well))
    plt.clf()
    plt.close()






