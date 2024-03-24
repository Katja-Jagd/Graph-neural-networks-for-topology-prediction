import proteinworkshop
import pickle
import graphein.protein.tensor as gpt
import torch
import torchdrug
from typing import Optional, Set, Union
import torch.nn as nn
import torch.nn.functional as F
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_scatter import scatter_add
from proteinworkshop.models.graph_encoders.layers import gear_net
from proteinworkshop.models.utils import get_aggregation
from proteinworkshop.types import EncoderOutput
import json
import pickle
from torch_geometric.data import Dataset
import glob
from graphein.protein.utils import download_alphafold_structure
from torch_geometric.loader import DataLoader
import os
import importlib

class GearNetDataset(Dataset):    
    
    def __init__(self, root, protein_list, data_dict, transform=None,pre_transform=None, pre_filter=None):
        self.protein_list = protein_list
        #print("Protein list in __init__:", self.protein_list)
        self.data_dict = data_dict
        self.root = root # path to directory containing data 
        self.missing_proteins = []
        super().__init__(root, transform, pre_transform, pre_filter)
    
        #UNDERSTAND TRANSFORM 
    
    @property
    def raw_file_names(self):
        """Names of raw files in the dataset"""
        # Creating a list of full paths to all downloaded .pdb files
        #all_downloads = glob.glob(os.path.join(self.raw_dir, f'*.pdb'))
        # Checking which files from the protein list is downloaded  
        #res = list(set([sub1 for ele1 in  all_downloads for sub1 in self.protein_list if sub1 in ele1]))
        # Updating the protein list to only contain downloaded files  
        #self.protein_list = res
        
        #return [f"{pid}.pdb" for pid in res] 
        return [f"{pid}.pdb" for pid in self.protein_list] 
    
    
    @property
    def processed_file_names(self):
        """Names of processed files to look for"""
        #tmp = [f"{pid}.pt" for pid in self.updated_protein_list]
        #print(len(tmp))
        #return tmp 
        return [f"{pid}.pt" for pid in self.protein_list] 
        #return [item for item in self.protein_list if item not in self.missing_proteins]
    
    def download(self):
        """Download the PDB files from AlphaFold"""
        #print("Protein list in download:", self.protein_list)
        for protein in self.protein_list:
            if os.path.exists(os.path.join(self.raw_dir, f'{protein}.pdb')):
                #print(f'\n{protein} is already downloaded')
                continue  
            else:
                download_alphafold_structure(protein, version=4,out_dir = self.raw_dir, aligned_score=False)
                # Check if proteins could not be downloaded 
                check = os.path.join(self.raw_dir, f"{protein}.pdb")
                if not os.path.isfile(check):
                    #print(f"Warning: The file for protein ID '{protein}' was not downloaded.")
                    self.missing_proteins.append(protein)
        
        #Update protein list, based on succesfull downloads
        self.protein_list = [item for item in self.protein_list if item not in self.missing_proteins]
        return
    
    
    def process(self):
        """Processes protein structures from pdb files into PyTorch Geometric Data"""
                
        # raw_paths inhereted from Dataset 
        
        for i, raw_path in enumerate(self.raw_paths):

            # Check if the datapoints have already been processed, if yes no processing if performed
            tmp_pdb_file = os.path.join(self.raw_dir, f'{self.protein_list[i]}.pdb')
            tmp_pt_file = os.path.join(self.processed_dir, f'{self.protein_list[i]}.pt')
            if os.path.isfile(tmp_pt_file) or os.path.isfile(tmp_pdb_file) == False: 
                print(f'{i}: Skipped processing {self.protein_list[i]}')
                continue

            # Preprocess data object if it has not been preprocessed already 
            else:
                print(f'{i}: Processing {self.protein_list[i]}')
                protein = gpt.Protein().from_pdb_file(os.path.join(self.raw_dir, f'{self.protein_list[i]}.pdb'))

                # Adding required batch attributes for GearNet 
                # https://proteins.sh/modules/proteinworkshop.models.html#proteinworkshop.models.graph_encoders.gear_net.GearNet

                # Node features, one-hot of residue types
                ##protein.x = protein.amino_acid_one_hot()
                
                # Node positions, coordinates for C-alpha atoms 
                ##protein.pos = protein.alpha_carbon(cache="ca")

                # All edges, Radius, KNN  (maybe sequential later)
                ##eps = protein.edges("eps_10")
                ##knn = protein.edges("knn_10")
                ##protein.edge_index = torch.cat((eps, knn), dim=1)  
                
                # Types of edges (Indicates which edges is where in the edge_index attribute)
                ##protein.edge_type = torch.cat((torch.zeros(eps.size(1), dtype=torch.long), 
                ##                               torch.ones(knn.size(1), dtype=torch.long))).view(-1,1)
                # Number of edge types
                #protein.num_relation = torch.tensor(2)
                #protein.num_relation = 2
                
                # USING INBUILT FUNCTION IN GEARNET TO GET EDGE FEATURES/ATTRIBUTES
                # Random initilization of GearNet, num_relations is not random(knn+eps)
                ##gn = GearNet(input_dim=20, num_relation=2, num_layers=3, emb_dim=64, short_cut=True, concat_hidden=False, batch_norm=True, num_angle_bin=10, activation="relu", pool="sum")
                ##protein.edge_attr = gn.gear_net_edge_features(protein)
                
                # OWN IMPLEMENTATION OF EDGE ATTRIBUTES
                # Edge attributes, concat of 5 things (App.C1 https://arxiv.org/pdf/2203.06125.pdf)
                #one_hot_r_type = protein.amino_acid_one_hot()

                #f_i = one_hot_r_type[protein.edge_index[0,:]] #1
                #f_j = one_hot_r_type[protein.edge_index[1,:]] #2
                #one_hot_e_type = F.one_hot(protein.edge_type,2) #3
                #sequential_dist = torch.abs(protein.edge_index[0] - protein.edge_index[1]).view(-1, 1) #4
                #spatial_dist = protein.edge_distances(protein.ca, protein.edge_index, 2).view(-1, 1) #5
                #protein.edge_attr = torch.cat((f_i, f_j, one_hot_e_type, sequential_dist, spatial_dist), dim=1)
                protein.seq_pos = torch.arange(protein.coords.shape[0]).unsqueeze(-1)
                protein.pos = protein.alpha_carbon(cache="ca")
    
                # Adding label sequence from dict to each dataobject 
                protein.label = self.data_dict[self.protein_list[i]]['labels']

                # Maybe add CV fold information here

                # Option to filter data out, e.g. only keep proteins smaller than 100 AA 
                if self.pre_filter is not None and not self.pre_filter(protein):
                    continue

                # Option to transform the data, e.g. normalization (MAYBE MOVE?? SMAE AS TRANSFORM HERE??)
                if self.pre_transform is not None:
                    protein = self.pre_transform(protein)

                # From graphein.protein.tensor.data.Protein to torch_geometric.data.data.Data
                protein = protein.to_data()
                
                # Save data object as .pt file 
                torch.save(protein, os.path.join(self.processed_dir, f'{self.protein_list[i]}.pt'))
                
    def len(self):
        """Returns length of data set (number of proteins)"""
        #tmp = len(self.protein_list)
        #print(f'From the len function: {tmp}')
        #return tmp
        return len(self.protein_list)

    def get(self, idx: int):
        """
        Returns Protein Data object for a given index

        :param idx: Index to retrieve.
        :type idx: int
        :return: Protein data object.
        """
        fname = f'{self.protein_list[idx]}.pt'
        return torch.load(os.path.join(self.processed_dir, fname))
    

        #if isinstance(idx, int):
        #    return torch.load(
        #        os.path.join(
        #            self.processed_dir, f"{self.protein_list[idx]}.pt"
        #        )
        #    )
        #elif isinstance(idx, str):
            #try:
        #    return torch.load(
        #        os.path.join(self.processed_dir, f"{idx}.pt")
        #    )
            #except FileNotFoundError:
            #    files = os.listdir(self.processed_dir)
            #    similar_files = get_close_matches(         # HAVENT DEFINES GET_CLOSE_MATHCHES YET
            #        f"{idx}.pt", files, n=5, cutoff=0.7
            #    )
            #    log.error(
            #        f"File {idx}.pt not found. Did you mean: {similar_files}"
            #    )
    def check_indices_length(self):
        indices = self.indices()
        print(len(indices))
        return len(indices)
