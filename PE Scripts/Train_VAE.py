import os.path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from os.path import join
import umap
import sys

from cytoself.datamanager.opencell import DataManagerOpenCell
from cytoself.trainer.cytoselflite_trainer import CytoselfFullTrainer
from cytoself.analysis.analysis_opencell import AnalysisOpenCell
from cytoself.trainer.utils.plot_history import plot_history_cytoself


def Prepare_Cells_3Ch(_data):

    _data_out = np.zeros([len(_data), 3, _data.shape[-2], _data.shape[-1]])

    for i in tqdm(range(len(_data))):

        for j in range(3):
            _data[i, j] = (_data[i, j] - _data[i, j].min()) / (_data[i, j].max() - _data[i, j].min())

        _nuc = _data[i, 2] * _data[i, 1]
        _cyt = _data[i, 2] * (_nuc == 0)

        ch_0 = _data[i, 0] * _data[i, 2]
        ch_1 = _data[i, 0] * _nuc
        ch_2 = _data[i, 0] * _cyt

        _data_out[i, 0] = ch_0
        _data_out[i, 1] = ch_1
        _data_out[i, 2] = ch_2

    return _data_out

def Prepare_Cells_2Ch(_data):

    _data_out = np.zeros([len(_data), 2, _data.shape[-2], _data.shape[-1]])

    for i in tqdm(range(len(_data))):

        for j in range(3):
            _data[i, j] = (_data[i, j] - _data[i, j].min()) / (_data[i, j].max() - _data[i, j].min())

        _nuc = _data[i, 2] * _data[i, 1]
        # _cyt = _data[i, 2] * (_nuc == 0)

        # ch_0 = _data[i, 0] * _data[i, 2]
        ch_0 = _data[i, 0] * _data[i, 2]
        ch_1 = _data[i, 0] * _nuc

        _data_out[i, 0] = ch_0
        _data_out[i, 1] = ch_1

    return _data_out

def Plot_Reconstruction(_img, _trainer):

    _reconstructed = _trainer.infer_reconstruction(_img)

    sph = 5
    spw = 10

    plt.figure(figsize=(24, 12))
    for i in range(len(_img)):
        plt.subplot(sph, spw, i + 1)
        plt.imshow(_img[i, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(join(_trainer.savepath_dict['visualization'], 'original_images.png'), dpi=300)
    plt.clf()
    plt.close()

    plt.figure(figsize=(24, 12))
    for i in range(len(_reconstructed)):
        plt.subplot(sph, spw, i + 1)
        plt.imshow(_reconstructed[i, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(join(_trainer.savepath_dict['visualization'], 'reconstructed_images.png'), dpi=300)
    plt.clf()
    plt.close()


t_start=datetime.now()

# 0. Check GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device, torch.cuda.get_device_name())

# 1. Prepare Data
# data_path = '/mnt/isilon/shalemlab/OpticalSequencing/data/TLopt_10March2022_V3/datasets'

# Experiment 1
image_all = np.load('data_4_cropseq_1500cpg.npy')
label_all_raw = pd.read_csv('labels_4_cropseq_1500cpg.csv')

# Experiment 2
# image_all = np.load('data_4_lenti_1500cpg.npy')
# label_all_raw = pd.read_csv('labels_4_lenti_1500cpg.csv')

image_all = Prepare_Cells_3Ch(image_all)
label_all = label_all_raw['local'].to_numpy()
label_all = np.expand_dims(label_all, -1)

n_ch = image_all.shape[1]
n_img = image_all.shape[-1]

for ed in [64]:
    for en in [512]:

            batch_size = 256
            emb_dim = ed
            emb_num = en
            fc_output_idx = [2]
            vae_name = 'fc-2_4_crop_1500cpg'
            save_path = 'Run_' + vae_name + '_ED-' + str(ed) + '_EN-' + str(en)

            datamanager = DataManagerOpenCell('', [], data_split=(0.70, 0.15, 0.15), fov_col=None)
            datamanager.numpy_dataloader(image_all, label_all, batch_size=batch_size, num_workers=4)

            # 2. Create and train a cytoself model
            model_args = {'input_shape': (n_ch, n_img, n_img), 'emb_shapes': ((32, 32), (4, 4)), 'output_shape': (n_ch, n_img, n_img),
                'fc_output_idx': fc_output_idx, 'vq_args': {'num_embeddings': emb_num, 'embedding_dim': emb_dim},
                'num_class': len(datamanager.unique_labels), 'fc_input_type': 'vqindhist', 'fc_args': {'num_layers': 2, 'num_features': 1000}}

            train_args = {'lr': 1e-4, 'max_epoch': 100, 'reducelr_patience': 3, 'reducelr_increment': 0.1, 'earlystop_patience': 10}

            trainer = CytoselfFullTrainer(train_args, homepath=save_path, model_args=model_args, device=device)
            trainer.fit(datamanager)
            # trainer.fit(datamanager, tensorboard_path='tb_logs')

            # 2.1 Generate training history
            plot_history_cytoself(trainer.history, savepath=trainer.savepath_dict['visualization'])

            # 3. Analyze embeddings
            # analysis = AnalysisOpenCell(datamanager, trainer)

            # 2.2 Compare the reconstructed images as a sanity check
            img = next(iter(datamanager.test_loader))['image'].detach().cpu().numpy()
            torch.cuda.empty_cache()
            Plot_Reconstruction(img[:50], trainer)

            print('Embedding full data set')

            embeddings_all = np.zeros([len(image_all), ed, 4, 4])
            embeddings_vqvecind_1_all = np.zeros([len(image_all), en])
            embeddings_vqvecind_2_all = np.zeros([len(image_all), en])

            itr_idx = np.arange(0, len(image_all) + batch_size, batch_size, dtype=int)
            for i in tqdm(range(len(itr_idx) - 1)):
                img_batch = image_all[itr_idx[i]:itr_idx[i + 1]]

                embeddings_all[itr_idx[i]:itr_idx[i + 1]] = trainer.infer_embeddings(img_batch)
                embeddings_vqvecind_1_all[itr_idx[i]:itr_idx[i + 1]] = trainer.infer_embeddings(img_batch,
                                                                                                output_layer=f'vqindhist{1}')
                embeddings_vqvecind_2_all[itr_idx[i]:itr_idx[i + 1]] = trainer.infer_embeddings(img_batch,
                                                                                                output_layer=f'vqindhist{2}')

            np.save(join(save_path, 'Inferred_latent_' + vae_name + '_ED-' + str(ed) + '_EN-' + str(en) + '.npy'), embeddings_all)
            np.save(join(save_path, 'Inferred_vqindhist1_' + vae_name + '_ED-' + str(ed) + '_EN-' + str(en) + '.npy'), embeddings_vqvecind_1_all)
            np.save(join(save_path, 'Inferred_vqindhist2_' + vae_name + '_ED-' + str(ed) + '_EN-' + str(en) + '.npy'), embeddings_vqvecind_2_all)

            reducer = umap.UMAP(verbose=False, random_state=0)

            E_latent = reducer.fit_transform(embeddings_all.reshape(embeddings_all.shape[0], -1))
            np.save(join(save_path, 'Embedded_latent_' + vae_name + '_ED-' + str(ed) + '_EN-' + str(en) + '.npy'), E_latent)

            E_vqvec_1 = reducer.fit_transform(embeddings_vqvecind_1_all)
            np.save(join(save_path, 'Embedded_vqindhist1_' + vae_name + '_ED-' + str(ed) + '_EN-' + str(en) + '.npy'), E_vqvec_1)

            E_vqvec_2 = reducer.fit_transform(embeddings_vqvecind_2_all)
            np.save(join(save_path, 'Embedded_vqindhist2_' + vae_name + '_ED-' + str(ed) + '_EN-' + str(en) + '.npy'), E_vqvec_2)


t_end=datetime.now()
print('Time:', t_end-t_start, ' Start:', t_start, ' Finish:', t_end)