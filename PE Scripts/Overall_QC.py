import In_Situ_Functions as isf
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
import os, sys
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=int)
    args = parser.parse_args()
    n_job = args.job

    n_well = int(n_job)

    save_path = 'genotyping_results/well_' + str(n_well)
    nuc_save_path = os.path.join('segmented/10X/nucs/well_' + str(n_well))
    cell_save_path = os.path.join('segmented/10X/cells/well_' + str(n_well))

    if np.any(n_well == np.array([1, 2, 4], dtype=int)):
        df_lib = pd.read_csv('20230411_PE_library_LentiGuide-BC_mNG2_60_reference.csv')
    if np.any(n_well == np.array([3, 5, 6], dtype=int)):
        df_lib = pd.read_csv('20230411_PE_library_CROPseq_mNG2_60_reference.csv')

    onlyfiles = natsorted([f for f in os.listdir(os.path.join(save_path, 'Cell_Genotype')) if os.path.isfile(os.path.join(save_path, 'Cell_Genotype', f)) and f.endswith('.csv')])
    for file in tqdm(onlyfiles):
        n_tile = int(file.split('_')[3])
        # try:
        isf.QC.Tile_QC(n_tile, n_well, save_path, nuc_save_path, cell_save_path, n_top=5)
        # except:
        #     print('QC for ' + file + ' could not be calculated')

    df_cells_complete = isf.Concatenate_CSV(save_path, 'Cell_Genotype')
    df_cells_complete.to_csv(os.path.join(save_path, 'Cells_Genotyped_Complete_Well_' + str(n_well) + '.csv'))

    df_quality_score = isf.Concatenate_CSV(save_path, 'Reads')

    isf.QC.Print_QC_Report(df_cells_complete, df_lib, save_path)
    isf.QC.Quality_Score_Plots(df_quality_score, save_path)
    isf.QC.Plot_Rank(df_cells_complete, save_path)
    isf.QC.Plot_sgRNAs_In_Gene(df_cells_complete, save_path)
    isf.QC.Plot_sgRNAs_In_Intron(df_cells_complete, save_path)
    isf.QC.Plot_Introns_In_Gene(df_cells_complete, save_path)



