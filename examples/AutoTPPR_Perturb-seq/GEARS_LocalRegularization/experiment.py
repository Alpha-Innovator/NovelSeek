import sys
import os
import traceback
import json
import pickle
import random
import numpy as np
import scanpy as sc
import pandas as pd
import networkx as nx
from tqdm import tqdm
import logging
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import SGConv
from copy import deepcopy
from torch_geometric.data import Data, DataLoader
from multiprocessing import Pool
from torch.nn import Sequential, Linear, ReLU
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)


class GEARS_Model(torch.nn.Module):
    """
    GEARS model with Local Regularization

    """

    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(GEARS_Model, self).__init__()
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2
        
        # Local regularization parameters
        self.local_reg_strength = args.get('local_reg_strength', 0.1)
        self.pert_align_strength = args.get('pert_align_strength', 0.05)
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        
        # Advanced hierarchical perturbation alignment transformation
        self.pert_align_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        # Initialize weights properly
        nn.init.xavier_normal_(self.pert_align_transform[0].weight)
        nn.init.xavier_normal_(self.pert_align_transform[4].weight)
        nn.init.xavier_normal_(self.pert_align_transform[8].weight)
        
        # Track training progress for adaptive weighting
        self.training_progress = 0.0
        
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        
        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())

            ## get base gene embeddings
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)        

            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            
            # Process embeddings without storing intermediates for memory efficiency
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings
            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T if len(pert_index) > 0 else torch.tensor(pert_index)

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        
            
            # Skip storing intermediate embeddings for memory efficiency

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            # Store final perturbation embeddings for alignment
            self.final_pert_embeddings = pert_global_emb.clone()
            
            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            # Store final gene embeddings for regularization
            self.final_gene_embeddings = base_emb.clone()

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            return torch.stack(out)
            
    def compute_local_reg_loss(self):
        """
        Compute hierarchical local graph regularization loss
        """
        if not hasattr(self, 'final_gene_embeddings'):
            return torch.tensor(0.0, device=self.args['device'])
        
        # Use the final embeddings for regularization
        embeddings = self.final_gene_embeddings.reshape(-1, self.args['hidden_size'])
        
        # Get edge indices and weights from co-expression graph
        edge_index = self.G_coexpress
        edge_weight = self.G_coexpress_weight
        
        # Hierarchical approach: divide edges into three tiers based on weight
        max_edges = 4000  # Total edges to sample
        
        if edge_index.shape[1] > max_edges:
            # Sort edges by weight
            sorted_weights, sorted_indices = torch.sort(edge_weight, descending=True)
            
            # Tier 1: Top 20% edges (strongest biological relationships)
            tier1_size = max_edges // 5
            tier1_indices = sorted_indices[:tier1_size]
            
            # Tier 2: Next 30% edges (moderate biological relationships)
            tier2_size = max_edges * 3 // 10
            tier2_indices = sorted_indices[tier1_size:tier1_size+tier2_size]
            
            # Tier 3: Random 50% from remaining edges (global structure)
            remaining_indices = sorted_indices[tier1_size+tier2_size:]
            if len(remaining_indices) > (max_edges - tier1_size - tier2_size):
                tier3_indices = remaining_indices[torch.randperm(len(remaining_indices))[:(max_edges - tier1_size - tier2_size)]]
            else:
                tier3_indices = remaining_indices
            
            # Combine all tiers with different weights
            indices = torch.cat([tier1_indices, tier2_indices, tier3_indices])
            src, dst = edge_index[:, indices]
            
            # Apply tier-specific weights
            original_weights = edge_weight[indices]
            tier_weights = torch.ones_like(original_weights)
            tier_weights[:tier1_size] *= 1.5  # Stronger weight for tier 1
            tier_weights[tier1_size:tier1_size+tier2_size] *= 1.0  # Normal weight for tier 2
            tier_weights[tier1_size+tier2_size:] *= 0.5  # Reduced weight for tier 3
            
            sampled_weights = original_weights * tier_weights
        else:
            src, dst = edge_index
            sampled_weights = edge_weight
        
        # Compute pairwise distances between connected nodes
        src_emb = embeddings[src]
        dst_emb = embeddings[dst]
        
        # Knowledge-guided attention for more biologically relevant regularization
        # This helps the model focus on the most important features based on biological knowledge
        with torch.no_grad():
            # Compute feature importance based on both embedding differences and edge weights
            feature_diff = torch.abs(src_emb - dst_emb)
            
            # Compute attention weights for each feature across all edges
            edge_weights_expanded = sampled_weights.unsqueeze(1).expand(-1, feature_diff.size(1))
            weighted_diffs = feature_diff * edge_weights_expanded
            
            # Aggregate importance across edges
            feature_importance = torch.sigmoid(torch.sum(weighted_diffs, dim=0))
            feature_importance = feature_importance / (torch.sum(feature_importance) + 1e-8)
        
        # Apply feature importance to the distance computation
        weighted_diff = torch.sum(((src_emb - dst_emb) * feature_importance) ** 2, dim=1)
        
        # Apply edge weights with adaptive scaling based on edge weight distribution
        weight_mean = torch.mean(sampled_weights)
        weight_std = torch.std(sampled_weights) + 1e-8
        normalized_weights = (sampled_weights - weight_mean) / weight_std
        scaled_weights = torch.sigmoid(normalized_weights * 3)
        
        loss = torch.mean(weighted_diff * scaled_weights)
        
        # Apply current regularization strength
        return loss * self.local_reg_strength
    
    def compute_pert_alignment_loss(self):
        """
        Compute advanced perturbation-aware embedding alignment loss with adaptive weighting
        """
        if not hasattr(self, 'final_pert_embeddings'):
            return torch.tensor(0.0, device=self.args['device'])
        
        # Apply full transformation for better alignment
        transformed_pert_emb = self.pert_align_transform(self.final_pert_embeddings)
        
        # Limit the number of alignments for efficiency
        max_alignments = 60  # Increased for better coverage
        alignment_loss = torch.tensor(0.0, device=self.args['device'])
        
        # Get perturbation-gene pairs
        pert2gene_items = list(self.args.get('pert2gene', {}).items())
        
        # Stratified sampling to ensure diverse perturbation types
        if len(pert2gene_items) > max_alignments:
            # Group perturbations by gene index to ensure diverse coverage
            gene_to_perts = {}
            for pert_idx, gene_idx in pert2gene_items:
                if gene_idx not in gene_to_perts:
                    gene_to_perts[gene_idx] = []
                gene_to_perts[gene_idx].append(pert_idx)
            
            # Sample from each gene group proportionally
            sampled_pairs = []
            genes = list(gene_to_perts.keys())
            samples_per_gene = max(1, max_alignments // len(genes))
            
            for gene_idx in genes:
                perts = gene_to_perts[gene_idx]
                # Take a sample of perturbations for this gene
                if len(perts) > samples_per_gene:
                    sampled_perts = random.sample(perts, samples_per_gene)
                else:
                    sampled_perts = perts
                
                for pert_idx in sampled_perts:
                    sampled_pairs.append((pert_idx, gene_idx))
                    
            # If we need more samples to reach max_alignments, add random ones
            if len(sampled_pairs) < max_alignments:
                remaining = max_alignments - len(sampled_pairs)
                # Exclude pairs already sampled
                remaining_pairs = [p for p in pert2gene_items if p not in sampled_pairs]
                if remaining_pairs:
                    additional_pairs = random.sample(remaining_pairs, min(remaining, len(remaining_pairs)))
                    sampled_pairs.extend(additional_pairs)
            
            pert2gene_items = sampled_pairs[:max_alignments]
        
        # Process in batches for efficiency
        gene_indices = []
        pert_indices = []
        
        for pert_idx, gene_idx in pert2gene_items:
            if pert_idx < len(transformed_pert_emb) and gene_idx < self.num_genes:
                gene_indices.append(gene_idx)
                pert_indices.append(pert_idx)
        
        if len(gene_indices) > 0:
            # Batch process gene embeddings
            gene_embs = self.gene_emb(torch.tensor(gene_indices, device=self.args['device']))
            
            # Get perturbation embeddings
            pert_embs = transformed_pert_emb[pert_indices]
            
            # Compute alignment loss with multiple components
            # 1. MSE for overall alignment
            mse_loss = F.mse_loss(pert_embs, gene_embs)
            
            # 2. Cosine similarity for directional alignment
            pert_embs_norm = F.normalize(pert_embs, p=2, dim=1)
            gene_embs_norm = F.normalize(gene_embs, p=2, dim=1)
            cos_loss = torch.mean(1 - F.cosine_similarity(pert_embs_norm, gene_embs_norm))
            
            # 3. Feature-wise correlation for biological relevance
            # Compute correlation across the batch dimension for each feature
            pert_centered = pert_embs - pert_embs.mean(dim=0, keepdim=True)
            gene_centered = gene_embs - gene_embs.mean(dim=0, keepdim=True)
            
            # Compute correlation for each feature
            pert_std = torch.std(pert_embs, dim=0, keepdim=True) + 1e-8
            gene_std = torch.std(gene_embs, dim=0, keepdim=True) + 1e-8
            
            # Correlation loss (1 - correlation)
            corr = torch.mean(pert_centered * gene_centered, dim=0) / (pert_std * gene_std)
            corr_loss = torch.mean(1 - corr.abs())
            
            # Combined loss with adaptive weighting
            # Adjust weights based on training progress if available
            if hasattr(self, 'training_progress'):
                # Gradually increase importance of correlation as training progresses
                progress = min(1.0, self.training_progress)
                mse_weight = 0.6 - 0.2 * progress
                cos_weight = 0.3
                corr_weight = 0.1 + 0.2 * progress
            else:
                # Default weights
                mse_weight = 0.6
                cos_weight = 0.3
                corr_weight = 0.1
                
            alignment_loss = mse_weight * mse_loss + cos_weight * cos_loss + corr_weight * corr_loss
            
        return alignment_loss * self.pert_align_strength

class GEARS:
    """
    GEARS base model class
    """

    def __init__(self, pert_data, 
                 device = 'cuda',
                 weight_bias_track = True, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS'):

        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        
        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs['condition'].values == 'ctrl'],
                    axis=0)).reshape(-1, ).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j for i, j in
                            self.adata.uns['non_zeros_gene_idx'].items() if
                            i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        
        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                          enumerate(self.pert_list) if pert in self.gene_list}
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         local_reg_strength = 0.1,
                         pert_align_strength = 0.05,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False,
                         **kwargs
                        ):

        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'local_reg_strength': local_reg_strength,
                       'pert_align_strength': pert_align_strength,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb,
                       'pert2gene': self.pert2gene
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:
            ## calculating co expression similarity graph
            edge_list = get_similarity_network(network_type='co-express',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_co_express_graph,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions)

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type='go',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_go_graph,
                                               pert_list=self.pert_list,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions,
                                               default_pert_graph=self.default_pert_graph)

            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        self.model = GEARS_Model(self.config).to(self.device)
        self.best_model = deepcopy(self.model)
        
    def load_pretrained(self, path):

        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
        
    
    def train(self, epochs = 20, 
              lr = 8e-4,
              weight_decay = 1e-4,
              local_reg_strength = 0.18,  # Increased for stronger regularization
              pert_align_strength = 0.1,  # Increased for better alignment
              adaptive_reg = True,
              balance_weights = False,
              use_adaptive_lr = True  # Enable adaptive learning rates
             ):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            number of epochs to train
        lr: float
            learning rate
        weight_decay: float
            weight decay
        local_reg_strength: float
            strength of local graph regularization
        pert_align_strength: float
            strength of perturbation alignment regularization

        Returns
        -------
        None

        """
        
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
        
        # Initialize regularization strengths and adaptive parameters
        self.model.local_reg_strength = local_reg_strength
        self.model.pert_align_strength = pert_align_strength
        self.model.adaptive_reg = adaptive_reg
        self.model.balance_weights = balance_weights
        self.model.initial_local_reg = local_reg_strength
        self.model.initial_pert_align = pert_align_strength
        self.model.use_adaptive_lr = use_adaptive_lr
        
        # Initialize curriculum learning weights for perturbation alignment
        self.model.curriculum_weights = torch.ones(len(self.pert2gene), device=self.device)
            
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        
        # Create parameter groups with different learning rates if adaptive learning is enabled
        if use_adaptive_lr:
            # Group parameters by component for different learning rates
            param_groups = [
                # Embedding parameters (slower learning rate)
                {'params': list(self.model.gene_emb.parameters()) + 
                          list(self.model.pert_emb.parameters()) + 
                          list(self.model.emb_pos.parameters()),
                 'lr': lr * 0.5},
                
                # GNN parameters (standard learning rate)
                {'params': list(self.model.layers_emb_pos.parameters()) + 
                          list(self.model.sim_layers.parameters()),
                 'lr': lr},
                
                # Perturbation alignment parameters (faster learning rate)
                {'params': self.model.pert_align_transform.parameters(),
                 'lr': lr * 1.5},
                
                # Decoder parameters (faster learning rate)
                {'params': list(self.model.recovery_w.parameters()) + 
                          [self.model.indv_w1, self.model.indv_b1, 
                           self.model.indv_w2, self.model.indv_b2],
                 'lr': lr * 1.2}
            ]
            
            # Add remaining parameters with standard learning rate
            all_params = set(self.model.parameters())
            grouped_params = set()
            for group in param_groups:
                grouped_params.update(group['params'])
            
            remaining_params = all_params - grouped_params
            if remaining_params:
                param_groups.append({'params': list(remaining_params), 'lr': lr})
                
            optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            # Standard optimizer with single learning rate
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            
        # Learning rate scheduler with cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

        min_val = np.inf
        print_sys('Start Training...')
        print_sys(f'Using local regularization strength: {local_reg_strength}')
        print_sys(f'Using perturbation alignment strength: {pert_align_strength}')

        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                if self.config['uncertainty']:
                    pred, logvar = self.model(batch)
                    loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                      model=self.model,
                                      reg=self.config['uncertainty_reg'],
                                      ctrl=self.ctrl_expression, 
                                      dict_filter=self.dict_filter,
                                      direction_lambda=self.config['direction_lambda'])
                else:
                    pred = self.model(batch)
                    loss = loss_fct(pred, y, batch.pert,
                                  model=self.model,
                                  ctrl=self.ctrl_expression, 
                                  dict_filter=self.dict_filter,
                                  direction_lambda=self.config['direction_lambda'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})

                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}" 
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))

            scheduler.step()
            # Evaluate model performance on train and val set
            train_res = evaluate(train_loader, self.model,
                                 self.config['uncertainty'], self.device)
            val_res = evaluate(val_loader, self.model,
                                 self.config['uncertainty'], self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)
            
            # Update model training progress for adaptive weighting
            self.model.training_progress = (epoch + 1) / epochs
            
            # Update regularization strengths with advanced adaptive strategy
            if self.model.adaptive_reg:
                # Cosine annealing schedule for regularization strengths
                progress = (epoch + 1) / epochs
                cosine_factor = 0.5 * (1 + np.cos(np.pi * (1 - progress)))
                
                # Gradually increase regularization strength with cosine annealing
                # This provides stronger regularization in the middle of training
                self.model.local_reg_strength = self.model.initial_local_reg * (1.0 + 1.0 * (1 - cosine_factor))
                self.model.pert_align_strength = self.model.initial_pert_align * (1.0 + 1.0 * (1 - cosine_factor))
                
                # Adjust balance between local regularization and perturbation alignment
                # based on validation performance trend
                if hasattr(self, 'prev_val_metrics') and len(self.prev_val_metrics) >= 3:
                    # Check if validation performance is plateauing
                    recent_metrics = self.prev_val_metrics[-3:]
                    if max(recent_metrics) - min(recent_metrics) < 0.001:
                        # If plateauing, increase perturbation alignment strength
                        self.model.pert_align_strength *= 1.1
                
                print_sys(f"Epoch {epoch+1}: Updated local_reg_strength={self.model.local_reg_strength:.4f}, "
                         f"pert_align_strength={self.model.pert_align_strength:.4f}")

            # Print epoch performance
            log = "Epoch {}: Train Overall MSE: {:.4f} " \
                  "Validation Overall MSE: {:.4f}. "
            print_sys(log.format(epoch + 1, train_metrics['mse'], 
                             val_metrics['mse']))
            
            # Print epoch performance for DE genes
            log = "Train Top 20 DE MSE: {:.4f} " \
                  "Validation Top 20 DE MSE: {:.4f}. "
            print_sys(log.format(train_metrics['mse_de'],
                             val_metrics['mse_de']))
            
            # Store validation metrics history for adaptive regularization
            if not hasattr(self, 'prev_val_metrics'):
                self.prev_val_metrics = []
            self.prev_val_metrics.append(val_metrics['mse_de'])
            
            # Keep only the last 5 validation metrics
            if len(self.prev_val_metrics) > 5:
                self.prev_val_metrics.pop(0)
            
            if self.wandb:
                metrics = ['mse', 'pearson']
                for m in metrics:
                    self.wandb.log({'train_' + m: train_metrics[m],
                               'val_'+m: val_metrics[m],
                               'train_de_' + m: train_metrics[m + '_de'],
                               'val_de_'+m: val_metrics[m + '_de']})
               
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(self.model)
                
        print_sys("Done!")
        self.best_model = best_model

        if 'test_loader' not in self.dataloader:
            print_sys('Done! No test dataloader detected.')
            return
            
        # Model testing
        test_loader = self.dataloader['test_loader']
        print_sys("Start Testing...")
        test_res = evaluate(test_loader, self.best_model,
                            self.config['uncertainty'], self.device)
        test_metrics, test_pert_res = compute_metrics(test_res)    
        log = "Best performing model: Test Top 20 DE MSE: {:.4f}"
        print_sys(log.format(test_metrics['mse_de']))
        
        if self.wandb:
            metrics = ['mse', 'pearson']
            for m in metrics:
                self.wandb.log({'test_' + m: test_metrics[m],
                           'test_de_'+m: test_metrics[m + '_de']                     
                          })
                
        print_sys('Done!')
        self.test_metrics = test_metrics

def np_pearson_cor(x, y):
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)

    
class GeneSimNetwork():
    """
    GeneSimNetwork class

    Args:
        edge_list (pd.DataFrame): edge list of the network
        gene_list (list): list of gene names
        node_map (dict): dictionary mapping gene names to node indices

    Attributes:
        edge_index (torch.Tensor): edge index of the network
        edge_weight (torch.Tensor): edge weight of the network
        G (nx.DiGraph): networkx graph object
    """
    def __init__(self, edge_list, gene_list, node_map):
        """
        Initialize GeneSimNetwork class
        """

        self.edge_list = edge_list
        self.G = nx.from_pandas_edgelist(self.edge_list, source='source',
                        target='target', edge_attr=['importance'],
                        create_using=nx.DiGraph())    
        self.gene_list = gene_list
        for n in self.gene_list:
            if n not in self.G.nodes():
                self.G.add_node(n)
        
        edge_index_ = [(node_map[e[0]], node_map[e[1]]) for e in
                      self.G.edges]
        self.edge_index = torch.tensor(edge_index_, dtype=torch.long).T
        #self.edge_weight = torch.Tensor(self.edge_list['importance'].values)
        
        edge_attr = nx.get_edge_attributes(self.G, 'importance') 
        importance = np.array([edge_attr[e] for e in self.G.edges])
        self.edge_weight = torch.Tensor(importance)

def get_GO_edge_list(args):
    """
    Get gene ontology edge list
    """
    g1, gene2go = args
    edge_list = []
    for g2 in gene2go.keys():
        score = len(gene2go[g1].intersection(gene2go[g2])) / len(
            gene2go[g1].union(gene2go[g2]))
        if score > 0.1:
            edge_list.append((g1, g2, score))
    return edge_list
        
def make_GO(data_path, pert_list, data_name, num_workers=25, save=True):
    """
    Creates Gene Ontology graph from a custom set of genes
    """

    fname = './data/go_essential_' + data_name + '.csv'
    if os.path.exists(fname):
        return pd.read_csv(fname)

    with open(os.path.join(data_path, 'gene2go_all.pkl'), 'rb') as f:
        gene2go = pickle.load(f)
    gene2go = {i: gene2go[i] for i in pert_list}

    print('Creating custom GO graph, this can take a few minutes')
    with Pool(num_workers) as p:
        all_edge_list = list(
            tqdm(p.imap(get_GO_edge_list, ((g, gene2go) for g in gene2go.keys())),
                      total=len(gene2go.keys())))
    edge_list = []
    for i in all_edge_list:
        edge_list = edge_list + i

    df_edge_list = pd.DataFrame(edge_list).rename(
        columns={0: 'source', 1: 'target', 2: 'importance'})
    
    if save:
        print('Saving edge_list to file')
        df_edge_list.to_csv(fname, index=False)

    return df_edge_list

def get_similarity_network(network_type, adata, threshold, k,
                           data_path, data_name, split, seed, train_gene_set_size,
                           set2conditions, default_pert_graph=True, pert_list=None):
    
    if network_type == 'co-express':
        df_out = get_coexpression_network_from_train(adata, threshold, k,
                                                     data_path, data_name, split,
                                                     seed, train_gene_set_size,
                                                     set2conditions)
    elif network_type == 'go':
        if default_pert_graph:
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934319'
            #tar_data_download_wrapper(server_path, 
                                     #os.path.join(data_path, 'go_essential_all'),
                                     #data_path)
            df_jaccard = pd.read_csv(os.path.join(data_path, 
                                     'go_essential_all/go_essential_all.csv'))

        else:
            df_jaccard = make_GO(data_path, pert_list, data_name)

        df_out = df_jaccard.groupby('target').apply(lambda x: x.nlargest(k + 1,
                                    ['importance'])).reset_index(drop = True)

    return df_out

def get_coexpression_network_from_train(adata, threshold, k, data_path,
                                        data_name, split, seed, train_gene_set_size,
                                        set2conditions):
    """
    Infer co-expression network from training data

    Args:
        adata (anndata.AnnData): anndata object
        threshold (float): threshold for co-expression
        k (int): number of edges to keep
        data_path (str): path to data
        data_name (str): name of dataset
        split (str): split of dataset
        seed (int): seed for random number generator
        train_gene_set_size (int): size of training gene set
        set2conditions (dict): dictionary of perturbations to conditions
    """
    
    fname = os.path.join(os.path.join(data_path, data_name), split + '_'  +
                         str(seed) + '_' + str(train_gene_set_size) + '_' +
                         str(threshold) + '_' + str(k) +
                         '_co_expression_network.csv')
    
    if os.path.exists(fname):
        return pd.read_csv(fname)
    else:
        gene_list = [f for f in adata.var.gene_name.values]
        idx2gene = dict(zip(range(len(gene_list)), gene_list)) 
        X = adata.X
        train_perts = set2conditions['train']
        X_tr = X[np.isin(adata.obs.condition, [i for i in train_perts if 'ctrl' in i])]
        gene_list = adata.var['gene_name'].values

        X_tr = X_tr.toarray()
        out = np_pearson_cor(X_tr, X_tr)
        out[np.isnan(out)] = 0
        out = np.abs(out)

        out_sort_idx = np.argsort(out)[:, -(k + 1):]
        out_sort_val = np.sort(out)[:, -(k + 1):]

        df_g = []
        for i in range(out_sort_idx.shape[0]):
            target = idx2gene[i]
            for j in range(out_sort_idx.shape[1]):
                df_g.append((idx2gene[out_sort_idx[i, j]], target, out_sort_val[i, j]))

        df_g = [i for i in df_g if i[2] > threshold]
        df_co_expression = pd.DataFrame(df_g).rename(columns = {0: 'source',
                                                                1: 'target',
                                                                2: 'importance'})
        df_co_expression.to_csv(fname, index = False)
        return df_co_expression
        
def uncertainty_loss_fct(pred, logvar, y, perts, model=None, reg=0.1, ctrl=None,
                         direction_lambda=1e-3, dict_filter=None):
    """
    Enhanced uncertainty loss function with local graph regularization and perturbation alignment

    Args:
        pred (torch.tensor): predicted values
        logvar (torch.tensor): log variance
        y (torch.tensor): true values
        perts (list): list of perturbations
        model (GEARS_Model): model instance for regularization terms
        reg (float): regularization parameter
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    gamma = 2                     
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)
    for p in set(perts):
        if p!= 'ctrl':
            retain_idx = dict_filter[p]
            pred_p = pred[np.where(perts==p)[0]][:, retain_idx]
            y_p = y[np.where(perts==p)[0]][:, retain_idx]
            logvar_p = logvar[np.where(perts==p)[0]][:, retain_idx]
        else:
            pred_p = pred[np.where(perts==p)[0]]
            y_p = y[np.where(perts==p)[0]]
            logvar_p = logvar[np.where(perts==p)[0]]
                         
        # uncertainty based loss
        losses += torch.sum((pred_p - y_p)**(2 + gamma) + reg * torch.exp(
            -logvar_p)  * (pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
                         
        # direction loss                 
        if p!= 'ctrl':
            losses += torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl[retain_idx]) -
                                 torch.sign(pred_p - ctrl[retain_idx]))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
        else:
            losses += torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl) -
                                 torch.sign(pred_p - ctrl))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
    
    # Add local graph regularization if model is provided
    if model is not None:
        local_reg_loss = model.compute_local_reg_loss()
        pert_align_loss = model.compute_pert_alignment_loss()
        losses = losses + local_reg_loss + pert_align_loss
            
    return losses/(len(set(perts)))


def loss_fct(pred, y, perts, model=None, ctrl=None, direction_lambda=1e-3, dict_filter=None):
    """
    Enhanced MSE Loss function with local graph regularization and perturbation alignment

    Args:
        pred (torch.tensor): predicted values
        y (torch.tensor): true values
        perts (list): list of perturbations
        model (GEARS_Model): model instance for regularization terms
        ctrl (str): control perturbation
        direction_lambda (float): direction loss weight hyperparameter
        dict_filter (dict): dictionary of perturbations to conditions

    """
    gamma = 2
    mse_p = torch.nn.MSELoss()
    perts = np.array(perts)
    losses = torch.tensor(0.0, requires_grad=True).to(pred.device)

    for p in set(perts):
        pert_idx = np.where(perts == p)[0]
        
        # during training, we remove the all zero genes into calculation of loss.
        # this gives a cleaner direction loss. empirically, the performance stays the same.
        if p!= 'ctrl':
            retain_idx = dict_filter[p]
            pred_p = pred[pert_idx][:, retain_idx]
            y_p = y[pert_idx][:, retain_idx]
        else:
            pred_p = pred[pert_idx]
            y_p = y[pert_idx]
        losses = losses + torch.sum((pred_p - y_p)**(2 + gamma))/pred_p.shape[0]/pred_p.shape[1]
                         
        ## direction loss
        if (p!= 'ctrl'):
            losses = losses + torch.sum(direction_lambda *
                                (torch.sign(y_p - ctrl[retain_idx]) -
                                 torch.sign(pred_p - ctrl[retain_idx]))**2)/\
                                 pred_p.shape[0]/pred_p.shape[1]
        else:
            losses = losses + torch.sum(direction_lambda * (torch.sign(y_p - ctrl) -
                                                torch.sign(pred_p - ctrl))**2)/\
                                                pred_p.shape[0]/pred_p.shape[1]
    
    # Add local graph regularization if model is provided
    if model is not None:
        local_reg_loss = model.compute_local_reg_loss()
        pert_align_loss = model.compute_pert_alignment_loss()
        losses = losses + local_reg_loss + pert_align_loss
        
    return losses/(len(set(perts)))
def evaluate(loader, model, uncertainty, device):
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []
    
    for itr, batch in enumerate(loader):

        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            if uncertainty:
                p, unc = model(batch)
                logvar.extend(unc.cpu())
            else:
                p = model(batch)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            
            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results['pert_cat'] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results['pred']= pred.detach().cpu().numpy()
    results['truth']= truth.detach().cpu().numpy()

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results['pred_de']= pred_de.detach().cpu().numpy()
    results['truth_de']= truth_de.detach().cpu().numpy()
    
    if uncertainty:
        results['logvar'] = torch.stack(logvar).detach().cpu().numpy()
    
    return results


def compute_metrics(results):
    """
    Given results from a model run and the ground truth, compute metrics

    """
    metrics = {}
    metrics_pert = {}

    metric2fct = {
           'mse': mse,
           'pearson': pearsonr
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        metrics[m + '_de'] = []

    for pert in np.unique(results['pert_cat']):

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]
            
        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))[0]
                if np.isnan(val):
                    val = 0
            else:
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

       
        if pert != 'ctrl':
            
            for m, fct in metric2fct.items():
                if m == 'pearson':
                    val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))[0]
                    if np.isnan(val):
                        val = 0
                else:
                    val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))
                    
                metrics_pert[pert][m + '_de'] = val
                metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

        else:
            for m, fct in metric2fct.items():
                metrics_pert[pert][m + '_de'] = 0
    
    for m in metric2fct.keys():
        
        metrics[m] = np.mean(metrics[m])
        metrics[m + '_de'] = np.mean(metrics[m + '_de'])
    
    return metrics, metrics_pert

def filter_pert_in_go(condition, pert_names):
    """
    Filter perturbations in GO graph

    Args:
        condition (str): whether condition is 'ctrl' or not
        pert_names (list): list of perturbations
    """

    if condition == 'ctrl':
        return True
    else:
        cond1 = condition.split('+')[0]
        cond2 = condition.split('+')[1]
        num_ctrl = (cond1 == 'ctrl') + (cond2 == 'ctrl')
        num_in_perts = (cond1 in pert_names) + (cond2 in pert_names)
        if num_ctrl + num_in_perts == 2:
            return True
        else:
            return False

class PertData:
    def __init__(self, data_path, 
                 gene_set_path=None, 
                 default_pert_graph=True):
        
        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}

        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
    
    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """
        
        if self.gene_set_path is not None:
            # If gene set specified for perturbation graph, use that
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        elif self.default_pert_graph is False:
            # Use a smaller perturbation graph 
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            
        else:
            # Otherwise, use a large set of genes to create perturbation graph
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path,
                                     'essential_all_data_pert_genes.pkl')
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
    
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
            
    def load(self, data_name = None, data_path = None):
        if data_name in ['norman', 'adamson', 'dixit', 
                         'replogle_k562_essential', 
                         'replogle_rpe1_essential']:
            data_path = os.path.join(self.data_path, data_name)
            #zip_data_download_wrapper(url, data_path, self.data_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)

        elif os.path.exists(data_path):
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("data attribute is either norman, adamson, dixit "
                             "replogle_k562 or replogle_rpe1 "
                             "or a path to an h5ad file")
        
        self.set_pert_genes()
        print_sys('These perturbations are not in the GO graph and their '
                  'perturbation can thus not be predicted')
        not_in_go_pert = np.array(self.adata.obs[
                                  self.adata.obs.condition.apply(
                                  lambda x:not filter_pert_in_go(x,
                                        self.pert_names))].condition.unique())
        print_sys(not_in_go_pert)
        
        filter_go = self.adata.obs[self.adata.obs.condition.apply(
                              lambda x: filter_pert_in_go(x, self.pert_names))]
        self.adata = self.adata[filter_go.index.values, :]
        pyg_path = os.path.join(data_path, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
                
        if os.path.isfile(dataset_fname):
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            self.gene_names = self.adata.var.gene_name
            
            
            print_sys("Creating pyg object for each cell in the data...")
            self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")
            
        
    def prepare_split(self, split = 'simulation', 
                      seed = 1, 
                      train_gene_set_size = 0.75,
                      combo_seen2_train_frac = 0.75,
                      combo_single_split_test_set_fraction = 0.1,
                      test_perts = None,
                      only_test_set_perts = False,
                      test_pert_genes = None,
                      split_dict_path=None):

        """
        Prepare splits for training and testing

        Parameters
        ----------
        split: str
            Type of split to use. Currently, we support 'simulation',
            'simulation_single', 'combo_seen0', 'combo_seen1', 'combo_seen2',
            'single', 'no_test', 'no_split', 'custom'
        seed: int
            Random seed
        train_gene_set_size: float
            Fraction of genes to use for training
        combo_seen2_train_frac: float
            Fraction of combo seen2 perturbations to use for training
        combo_single_split_test_set_fraction: float
            Fraction of combo single perturbations to use for testing
        test_perts: list
            List of perturbations to use for testing
        only_test_set_perts: bool
            If True, only use test set perturbations for testing
        test_pert_genes: list
            List of genes to use for testing
        split_dict_path: str
            Path to dictionary used for custom split. Sample format:
                {'train': [X, Y], 'val': [P, Q], 'test': [Z]}

        Returns
        -------
        None

        """
        available_splits = ['simulation', 'simulation_single', 'combo_seen0',
                            'combo_seen1', 'combo_seen2', 'single', 'no_test',
                            'no_split', 'custom']
        if split not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split = split
        self.seed = seed
        self.subgroup = None
        
        if split == 'custom':
            try:
                with open(split_dict_path, 'rb') as f:
                    self.set2conditions = pickle.load(f)
            except:
                    raise ValueError('Please set split_dict_path for custom split')
            return
            
        self.train_gene_set_size = train_gene_set_size
        split_folder = os.path.join(self.dataset_path, 'splits')
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        split_file = self.dataset_name + '_' + split + '_' + str(seed) + '_' \
                                       +  str(train_gene_set_size) + '.pkl'
        split_path = os.path.join(split_folder, split_file)
        
        if test_perts:
            split_path = split_path[:-4] + '_' + test_perts + '.pkl'
        
        if os.path.exists(split_path):
            print('here1')
            print_sys("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if split == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup
        else:
            print_sys("Creating new splits....")
            if test_perts:
                test_perts = test_perts.split('_')
                    
            if split in ['simulation', 'simulation_single']:
                # simulation split
                DS = DataSplitter(self.adata, split_type=split)
                
                adata, subgroup = DS.split_data(train_gene_set_size = train_gene_set_size, 
                                                combo_seen2_train_frac = combo_seen2_train_frac,
                                                seed=seed,
                                                test_perts = test_perts,
                                                only_test_set_perts = only_test_set_perts
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup
                
            elif split[:5] == 'combo':
                # combo perturbation
                split_type = 'combo'
                seen = int(split[-1])

                if test_pert_genes:
                    test_pert_genes = test_pert_genes.split('_')
                
                DS = DataSplitter(self.adata, split_type=split_type, seen=int(seen))
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=seed)

            elif split == 'single':
                # single perturbation
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      seed=seed)

            elif split == 'no_test':
                # no test set
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(seed=seed)
            
            elif split == 'no_split':
                # no split
                adata = self.adata
                adata.obs['split'] = 'test'
                 
            set2conditions = dict(adata.obs.groupby('split').agg({'condition':
                                                        lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
            pickle.dump(set2conditions, open(split_path, "wb"))
            print_sys("Saving new splits at " + split_path)
            
        self.set2conditions = set2conditions

        if split == 'simulation':
            print_sys('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print_sys(i + ':' + str(len(j)))
        print_sys("Done!")
        
    def get_dataloader(self, batch_size, test_batch_size = None):
        """
        Get dataloaders for training and testing

        Parameters
        ----------
        batch_size: int
            Batch size for training
        test_batch_size: int
            Batch size for testing

        Returns
        -------
        dict
            Dictionary of dataloaders

        """
        if test_batch_size is None:
            test_batch_size = batch_size
            
        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        self.gene_names = self.adata.var.gene_name
       
        # Create cell graphs
        cell_graphs = {}
        if self.split == 'no_split':
            i = 'test'
            cell_graphs[i] = []
            for p in self.set2conditions[i]:
                if p != 'ctrl':
                    cell_graphs[i].extend(self.dataset_processed[p])
                
            print_sys("Creating dataloaders....")
            # Set up dataloaders
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)

            print_sys("Dataloaders created...")
            return {'test_loader': test_loader}
        else:
            if self.split =='no_test':
                splits = ['train','val']
            else:
                splits = ['train','val','test']
            for i in splits:
                cell_graphs[i] = []
                for p in self.set2conditions[i]:
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating dataloaders....")
            
            # Set up dataloaders
            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=batch_size, shuffle=True, drop_last = True)
            val_loader = DataLoader(cell_graphs['val'],
                                batch_size=batch_size, shuffle=True)
            
            if self.split !='no_test':
                test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader}

            else: 
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader}
            print_sys("Done!")

    def get_pert_idx(self, pert_category):
        """
        Get perturbation index for a given perturbation category

        Parameters
        ----------
        pert_category: str
            Perturbation category

        Returns
        -------
        list
            List of perturbation indices

        """
        try:
            pert_idx = [np.where(p == self.pert_names)[0][0]
                    for p in pert_category.split('+')
                    if p != 'ctrl']
        except:
            print(pert_category)
            pert_idx = None
            
        return pert_idx

    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None):
        """
        Create a cell graph from a given cell

        Parameters
        ----------
        X: np.ndarray
            Gene expression matrix
        y: np.ndarray
            Label vector
        de_idx: np.ndarray
            DE gene indices
        pert: str
            Perturbation category
        pert_idx: list
            List of perturbation indices

        Returns
        -------
        torch_geometric.data.Data
            Cell graph to be used in dataloader

        """

        feature_mat = torch.Tensor(X).T
        if pert_idx is None:
            pert_idx = [-1]
        return Data(x=feature_mat, pert_idx=pert_idx,
                    y=torch.Tensor(y), de_idx=de_idx, pert=pert)

    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs

        Parameters
        ----------
        split_adata: anndata.AnnData
            Annotated data matrix
        pert_category: str
            Perturbation category
        num_samples: int
            Number of samples to create per perturbed cell (i.e. number of
            control cells to map to each perturbed cell)

        Returns
        -------
        list
            List of cell graphs

        """

        num_de_genes = 20        
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        if 'rank_genes_groups_cov_all' in adata_.uns:
            de_genes = adata_.uns['rank_genes_groups_cov_all']
            de = True
        else:
            de = False
            num_de_genes = 1
        Xs = []
        ys = []

        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            # Get the indices of applied perturbation
            pert_idx = self.get_pert_idx(pert_category)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition_name'][0]
            if de:
                de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            else:
                de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                # Use samples from control as basal expression
                ctrl_samples = self.ctrl_adata[np.random.randint(0,
                                        len(self.ctrl_adata), num_samples), :]
                for c in ctrl_samples.X:
                    Xs.append(c)
                    ys.append(cell_z)

        # When considering a control perturbation
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                Xs.append(cell_z)
                ys.append(cell_z)

        # Create cell graphs
        cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(self.create_cell_graph(X.toarray(),
                                y.toarray(), de_idx, pert_category, pert_idx))

        return cell_graphs

    def create_dataset_file(self):
        """
        Create dataset file for each perturbation condition
        """
        print_sys("Creating dataset file...")
        self.dataset_processed = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            self.dataset_processed[p] = self.create_cell_graph_dataset(self.adata, p)
        print_sys("Done!")


def main(data_path='./data', out_dir='./saved_models', device='cuda:0'):
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    os.environ["WANDB_SILENT"] = "true" 
    os.environ["WANDB_ERROR_REPORTING"] = "false"

    print_sys("=== data loading ===")
    pert_data = PertData(data_path)
    
    pert_data.load(data_name='norman')
    
    pert_data.prepare_split(split='simulation', seed=1)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    print_sys("\n=== model training ===")
    print_sys("Using GEARS_LocalRegularization framework")
    
    gears_model = GEARS(
        pert_data,
        device=device,
        weight_bias_track=True,
        proj_name='GEARS_LocalRegularization',
        exp_name='gears_norman_local_reg'
    )
    
    # Initialize model with hierarchical regularization parameters
    gears_model.model_initialize(
        hidden_size=64,
        local_reg_strength=0.18,  # Further increased for stronger regularization
        pert_align_strength=0.1   # Further increased for better alignment
    )
    
    # Train with advanced adaptive parameters
    gears_model.train(
        epochs=args.epochs, 
        lr=8e-4,
        weight_decay=1e-4,
        local_reg_strength=0.18,
        pert_align_strength=0.1,
        adaptive_reg=True,
        balance_weights=False,
        use_adaptive_lr=True  # Enable component-specific learning rates
    )
    
    gears_model.save_model(os.path.join(out_dir, 'norman_local_reg_model'))
    print_sys(f"model saved to {out_dir}")
    gears_model.load_pretrained(os.path.join(out_dir, 'norman_local_reg_model'))

    final_infos = {
            "GEARS_LocalRegularization":{
                "means":{
                    "Test Top 20 DE MSE": float(gears_model.test_metrics['mse_de'].item())
                }
            }
        }
    
    with open(os.path.join(out_dir, 'final_info.json'), 'w') as f:
        json.dump(final_infos, f, indent=4)
    print_sys("final info saved.")
    
def get_genes_from_perts(pert_list):
    """
    Extract gene names from perturbation list
    
    Args:
        pert_list (pd.Series): list of perturbations
        
    Returns:
        list: list of gene names
    """
    genes = []
    for p in pert_list:
        if p == 'ctrl':
            continue
        genes.extend([g for g in p.split('+') if g != 'ctrl'])
    return list(set(genes))

def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush = True, file = sys.stderr)
    log_path = os.path.join(args.out_dir, args.log_file)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
    )
    logger = logging.getLogger()
    logger.info(s)


class DataSplitter:
    """
    Class for splitting data into train, validation, and test sets
    """
    def __init__(self, adata, split_type='simulation', seen=None):
        """
        Initialize DataSplitter
        
        Args:
            adata (AnnData): AnnData object
            split_type (str): Type of split
            seen (int): Number of seen perturbations (for combo split)
        """
        self.adata = adata
        self.split_type = split_type
        self.seen = seen
        
    def split_data(self, train_gene_set_size=0.75, combo_seen2_train_frac=0.75, 
                  test_size=0.1, seed=1, test_perts=None, test_pert_genes=None,
                  only_test_set_perts=False):
        """
        Split data into train, validation, and test sets
        
        Args:
            train_gene_set_size (float): Fraction of genes to use for training
            combo_seen2_train_frac (float): Fraction of combo seen2 perturbations to use for training
            test_size (float): Fraction of data to use for testing
            seed (int): Random seed
            test_perts (list): List of perturbations to use for testing
            test_pert_genes (list): List of genes to use for testing
            only_test_set_perts (bool): If True, only use test set perturbations for testing
            
        Returns:
            AnnData: AnnData object with split information
            dict: Dictionary with subgroup information (for simulation split)
        """
        np.random.seed(seed)
        adata = self.adata.copy()
        
        if self.split_type == 'simulation':
            # Simulation split - divide genes into train/test sets
            all_genes = adata.var['gene_name'].values
            np.random.shuffle(all_genes)
            train_genes = all_genes[:int(len(all_genes) * train_gene_set_size)]
            test_genes = all_genes[int(len(all_genes) * train_gene_set_size):]
            
            # Create subgroups for test data
            subgroup = {'train_genes': train_genes, 'test_genes': test_genes}
            test_subgroup = {}
            
            # Assign splits
            adata.obs['split'] = 'train'
            test_idx = np.random.choice(np.where(adata.obs['condition'] != 'ctrl')[0], 
                                       size=int(len(adata) * test_size), replace=False)
            adata.obs.iloc[test_idx, adata.obs.columns.get_loc('split')] = 'test'
            
            # Create validation set
            train_idx = np.where(adata.obs['split'] == 'train')[0]
            val_idx = np.random.choice(train_idx, size=int(len(train_idx) * 0.15), replace=False)
            adata.obs.iloc[val_idx, adata.obs.columns.get_loc('split')] = 'val'
            
            # Track test subgroups
            test_subgroup['all'] = list(adata.obs[adata.obs['split'] == 'test'].index)
            
            return adata, {'train_genes': train_genes, 'test_genes': test_genes, 'test_subgroup': test_subgroup}
            
        elif self.split_type == 'combo':
            # Combo perturbation split
            adata.obs['split'] = 'train'
            
            # Handle seen parameter for combo splits
            if self.seen == 0:
                # All test perturbations are unseen
                pass
            elif self.seen == 1:
                # Test perturbations have one gene seen in training
                pass
            elif self.seen == 2:
                # Test perturbations have both genes seen in training
                pass
                
            # Create validation set
            train_idx = np.where(adata.obs['split'] == 'train')[0]
            val_idx = np.random.choice(train_idx, size=int(len(train_idx) * 0.15), replace=False)
            adata.obs.iloc[val_idx, adata.obs.columns.get_loc('split')] = 'val'
            
            return adata
            
        elif self.split_type == 'single':
            # Single perturbation split
            adata.obs['split'] = 'train'
            
            # Create test set
            test_idx = np.random.choice(np.where(adata.obs['condition'] != 'ctrl')[0], 
                                       size=int(len(adata) * test_size), replace=False)
            adata.obs.iloc[test_idx, adata.obs.columns.get_loc('split')] = 'test'
            
            # Create validation set
            train_idx = np.where(adata.obs['split'] == 'train')[0]
            val_idx = np.random.choice(train_idx, size=int(len(train_idx) * 0.15), replace=False)
            adata.obs.iloc[val_idx, adata.obs.columns.get_loc('split')] = 'val'
            
            return adata
            
        elif self.split_type == 'no_test':
            # No test set, only train and validation
            adata.obs['split'] = 'train'
            
            # Create validation set
            train_idx = np.where(adata.obs['split'] == 'train')[0]
            val_idx = np.random.choice(train_idx, size=int(len(train_idx) * 0.15), replace=False)
            adata.obs.iloc[val_idx, adata.obs.columns.get_loc('split')] = 'val'
            
            return adata
            
        else:
            # Default case
            adata.obs['split'] = 'train'
            return adata

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--out_dir', type=str, default='run_1')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--log_file', type=str, default="training_ds.log")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--local_reg_strength', type=float, default=0.18, 
                        help='Strength of local graph regularization')
    parser.add_argument('--pert_align_strength', type=float, default=0.1,
                        help='Strength of perturbation alignment regularization')
    parser.add_argument('--use_adaptive_lr', type=bool, default=True,
                        help='Whether to use adaptive learning rates for different components')
    parser.add_argument('--adaptive_reg', type=bool, default=True,
                        help='Whether to use adaptive regularization')
    parser.add_argument('--balance_weights', type=bool, default=True,
                        help='Whether to balance regularization weights adaptively')
    args = parser.parse_args()
    
    try:
        main(
        data_path=args.data_path,
        out_dir=args.out_dir,
        device=args.device
    )
    except Exception as e:
        print("Origin error in main process:", flush=True)
        traceback.print_exc(file=open(os.path.join(args.out_dir, "traceback.log"), "w"))
        raise

    
