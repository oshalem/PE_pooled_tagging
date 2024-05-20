import sys, os
from datetime import datetime
import matplotlib.pyplot as plt
import In_Situ_Functions as isf
import pandas as pd
import numpy as np
import argparse

if __name__ == '__main__':

    t_start=datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=int)
    args = parser.parse_args()
    n_job = args.job

    # shell script uses 5 digits, first is the well, and the following four are the tile
    # Example: Well 2, Tile 362, is 20362
    n_well = int(np.floor(n_job / 10000))
    n_tile = int(np.around(10000 * ((n_job / 10000) % 1)))

    save_path = 'genotyping_results/well_' + str(n_well)

    # Path to library file, connect nucleotide barcode to spacer
    if np.any(n_well == np.array([1, 2, 4], dtype=int)):
        df_lib = pd.read_csv('20230411_PE_library_LentiGuide-BC_mNG2_60_reference.csv')
    if np.any(n_well == np.array([3, 5, 6], dtype=int)):
        df_lib = pd.read_csv('20230411_PE_library_CROPseq_mNG2_60_reference.csv')

    isf.Manage_Save_Directories(save_path)

    data = isf.InSitu.Assemble_Data_From_ND2(n_tile, n_well, 'genotyping')
    data = np.concatenate((data[:, [-1]], data[:, :-1]), axis=1)

    maxed, peaks, _ = isf.InSitu.Find_Peaks(data, verbose=False)

    nucs = np.load(os.path.join('segmented', '10X', 'nucs', 'well_' + str(n_well), 'Seg_Nuc-Well_'+ str(n_well) + '_Tile_' + str(n_tile) + '.npy'))[1]
    cells = np.load(os.path.join('segmented', '10X', 'cells', 'well_' + str(n_well), 'Seg_Cells-Well_'+ str(n_well) + '_Tile_' + str(n_tile) + '.npy'))[1]

    df_reads = isf.InSitu.Call_Bases(cells, maxed, peaks, 100)
    df_reads = isf.Lookup.Reads_Q_lim(df_reads, 0.3)
    # df_reads_amb = isf.InSitu.Assign_Simple_Ambiguity(df_reads, lim=0.3)

    df_cell_genotype = isf.Lookup.Choose_Barcodes(df_reads, df_lib, nucs, cells)

    isf.Save(save_path, n_tile, n_well,  df_reads=df_reads, df_cell_genotype=df_cell_genotype)

    t_end=datetime.now()
    print('Time:',t_end-t_start, ' Start:', t_start, ' Finish:', t_end)