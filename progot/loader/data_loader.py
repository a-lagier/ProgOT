import torch
from scipy.sparse import coo_matrix, save_npz, load_npz
from scipy.io import mmwrite, mmread
import numpy as np
import os
import pandas as pd
from ..utils import select_top_one, perform_pca
from Wasserstein2Benchmark.src import map_benchmark as mbm


__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"{name} already registerd")
        __DATASET__[name] = cls
        return cls
    return wrapper

def get_dataset(name: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DATASET__[name](**kwargs)

@register_dataset('synthetic')
class SyntheticMix3ToMix10():

    def __init__(self, dataset_dir, d, train_sample, test_sample, **kwargs):

        self.dataset_dir = dataset_dir
        self.d = d
        self.n_train = train_sample
        self.n_test = test_sample
        self.benchmark = mbm.Mix3ToMix10Benchmark(d)

    def get_data(self):
        X_train = self.benchmark.input_sampler.sample(self.n_train)
        X_test = self.benchmark.input_sampler.sample(self.n_test)
        
        X_train.requires_grad_(True)
        X_test.requires_grad_(True)

        Y_train = self.benchmark.map_fwd(X_train, nograd=True)
        Y_test = self.benchmark.map_fwd(X_test, nograd=True)

        return X_train, X_test, Y_train, Y_test


@register_dataset('sci-plex')
class SciPlex():

    def __init__(self, dataset_dir, d, **kwargs):

        self.dataset_dir = os.path.join(dataset_dir, 'sci-plex/')
        self.preprocess_data_path = 'preprocessed_data/'
        
        self.genes_matrix_path = '3/GSM4150378_sciPlex3_A549_MCF7_K562_screen_UMI.count.matrix'
        self.cell_annotation_path = '3/GSM4150378_sciPlex3_A549_MCF7_K562_screen_cell.annotations.txt'
        self.pdata_path = '3/GSM4150378_sciPlex3_pData.txt'

        self.retained_drugs = ['Belinostat (PXD101)', 'Dacinostat (LAQ824)', 'Givinostat (ITF2357)', 'Hesperadin', 'Quisinostat (JNJ-26481585) 2HCl']
        self.drug_cell_dict = {drug: None for drug in self.retained_drugs}

        self.d = d

    def load_cell_genes(self):
        data = pd.read_csv(os.path.join(self.dataset_dir, self.genes_matrix_path), sep='\t', header=None, names=['gene_idx', 'cell_idx', 'count'])

        #TODO: filter genes
        genes_counts = data.groupby('gene_idx')['cell_idx'].nunique()
        valid_gene_idx = genes_counts[genes_counts >= 20].index

        cell_genes_counts = data.groupby('cell_idx')['gene_idx'].nunique()
        valid_cells_idx = cell_genes_counts[cell_genes_counts >= 20]

        self.valid_cells_idx = valid_cells_idx

        data = data[
            data['gene_idx'].isin(valid_gene_idx) &
            data['cell_idx'].isin(valid_cells_idx)
        ]

        row = data['cell_idx'].values
        col = data['gene_idx'].values
        val = data['count'].values

        return coo_matrix((val, (row - 1, col - 1)), (max(row) + 1, max(col) + 1)).tocsr()

    def load_cells_annotations(self):
        data = pd.read_csv(os.path.join(self.dataset_dir, self.cell_annotation_path), sep='\t', header=None)
        data = data[0]
        return pd.Series(range(len(data)), index=data.values)
    
    def load_cell_drug(self):
        data = pd.read_csv(os.path.join(self.dataset_dir, self.pdata_path), sep=' ')
        source_cells = data[data['vehicle'] == True]
        source_cells, main_cell_type = select_top_one(source_cells, 'cell_type')
        source_cells, main_time_point = select_top_one(source_cells, 'time_point')

        for drug in self.retained_drugs:
            target_cells = data[
                (data['product_name'] == drug) &
                (data['cell_type'] == main_cell_type) &
                (data['time_point'] == main_time_point)
            ]
            target_cells, _ = select_top_one(target_cells, 'dose')

            # TODO: write data back
            source_drug_cells = source_cells['cell']
            target_drug_cells = target_cells['cell']
            self.drug_cell_dict[drug] = (source_drug_cells, target_drug_cells)

    def preprocess_data(self):
        self.load_cell_drug()
        cell_to_idx = self.load_cells_annotations()
        cell_genes_matrix = self.load_cell_genes()
        for drug, c in self.drug_cell_dict.items():
            drug_name_file = drug.replace(' ', '.')

            s, t = c
            idx_s, idx_t = cell_to_idx[s.str.strip()], cell_to_idx[t.str.strip()]
            idx_s, idx_t = idx_s[idx_s.isin(self.valid_cells_idx)], idx_t[idx_t.isin(self.valid_cells_idx)]
            s_emb, t_emb = cell_genes_matrix[idx_s], cell_genes_matrix[idx_t]
            save_npz(os.path.join(self.dataset_dir, self.preprocess_data_path, f'{drug_name_file}_source.npz'), s_emb)
            save_npz(os.path.join(self.dataset_dir, self.preprocess_data_path, f'{drug_name_file}_target.npz'), t_emb)

    def get_data(self, drug):
        drug_name_file = drug.replace(' ', '.')

        source_emb = load_npz(os.path.join(self.dataset_dir, self.preprocess_data_path, f'{drug_name_file}_source.npz'))
        target_emb = load_npz(os.path.join(self.dataset_dir, self.preprocess_data_path, f'{drug_name_file}_target.npz'))

        source_emb = torch.tensor(source_emb.toarray())
        target_emb = torch.tensor(target_emb.toarray())

        source_emb = torch.log(source_emb + 1.)
        target_emb = torch.log(target_emb + 1.)

        source_emb = perform_pca(source_emb, self.d)
        target_emb = perform_pca(target_emb, self.d)
        
        return (source_emb, target_emb)

@register_dataset('4i')
class Fouri():

    def __init__(self, dataset_dir, **kwargs):
        import squidpy
        
        self.dataset_dir = dataset_dir

        self.data = squidpy.datasets.four_i(dataset_dir)
    
    def get_data(self):
        return self.data

# sp_loader = SciPlex("./sci-plex")
# sp_loader.preprocess_data()
# sp_loader.load_cell_genes()
# sp_loader.load_cell_drug()
# sp_loader.load_cells_annotations()
# import matplotlib.pyplot as plt
# s = SyntheticMix3ToMix10(2, 3)
# data = s.sample(30, std=0.5)
# m = s.mixtures
# print(data.shape)
# plt.scatter(data[:,0], data[:,1], c='b')
# plt.scatter(m[:,0], m[:,1], c='r')
# plt.grid()
# plt.savefig('synthetic.png')

