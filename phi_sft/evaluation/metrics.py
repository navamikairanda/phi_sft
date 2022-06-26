import torch
import os
import matplotlib.pyplot as plt
import pdb
import math
import numpy as np
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from geometry.point_cloud_io import load_pointclouds_from_dir
from geometry.mesh_io import load_meshes_from_dir

class Metrics():

    def __init__(self, args_expt, args_data, log, device):
        self.log = log
        self.device = device
        self.sequence_type = args_data['sequence_type']
        self.i_save = args_expt.getint('i_save')
        self.n_frames = args_expt.getint('n_frames')
        self.n_points_sample = args_data.getint('n_points_sample')
        if args_expt.getboolean('reload'): 
            self.surface_errors = torch.load(os.path.join(self.log_dir, args_expt['i_reload'], 'surface_errors.pt'))
        else: 
            if self.sequence_type == 'real':       
                self.surface_errors = {'chamfer_distance': [], 'lastframe_chamfer_distance': []} 
            elif self.sequence_type == 'synthetic':
                self.surface_errors = {'angular_error': [], '3d_error': []}   
        if self.sequence_type == 'real':
            self.gt_point_clouds, self.gt_point_clouds_lengths = load_pointclouds_from_dir(os.path.join(args_data['sequence_dir'], 'point_clouds'), args_expt.getint('n_frames'), device)
            
        elif self.sequence_type == 'synthetic':
            gt_meshes = load_meshes_from_dir(os.path.join(args_data['sequence_dir'], 'surfaces'), device=self.device)[:args_expt.getint('n_frames')]
            self.gt_meshes_verts = gt_meshes.verts_padded()
            self.gt_meshes_verts_normals = gt_meshes.verts_normals_padded()

    def compute_recon_errors(self, aligned_meshes):
        if self.sequence_type == 'real':
            self.compute_chamfer_distance_error(aligned_meshes)
        elif self.sequence_type == 'synthetic':
            self.compute_angular_error(aligned_meshes)
            self.compute_surface_error(aligned_meshes)
            
    def compute_chamfer_distance_error(self, recon_meshes): 
        """Computes chamfer distance (eq. 10 in paper) between ground-truth point cloud and points sampled from reconstructed mesh
        Args:
            recon_meshes: Reconstucted meshes after aligning to ground-truth
        Returns:
        """        
        recon_samples = sample_points_from_meshes(recon_meshes, self.n_points_sample)

        _chamfer_distance, _ = chamfer_distance(recon_samples, self.gt_point_clouds, y_lengths=self.gt_point_clouds_lengths, batch_reduction='mean', point_reduction='mean')
        _chamfer_distance_last_frame, _ = chamfer_distance(recon_samples[-1].unsqueeze(dim=0), self.gt_point_clouds[-1], y_lengths=self.gt_point_clouds_lengths[-1].unsqueeze(dim=0), batch_reduction='mean', point_reduction='mean')
        self.log.write('Chamfer distance with {} frames: {:05.6f}\n'.format(self.n_frames, _chamfer_distance)) 
        self.log.write('Chamfer distance for last frame: {:05.6f}\n'.format(_chamfer_distance_last_frame))        
        self.surface_errors['chamfer_distance'].append(_chamfer_distance) 
        self.surface_errors['lastframe_chamfer_distance'].append(_chamfer_distance_last_frame)
    
    def compute_surface_error(self, recon_meshes): 
        """Computes 3D error (eq. 1 in supplementary) between ground-truth meshes and reconstructed meshes
        Args:
            recon_meshes: Reconstucted meshes after aligning to ground-truth
        Returns:
            per_vertex_errors: [n_frames, n_vertices]. Reconstruction error computed per vertex, useful for visualisation
        """
        per_vertex_errors = (self.gt_meshes_verts - recon_meshes.verts_padded()).norm(dim=2) # [M, N]
        gt_meshes_verts_norm = self.gt_meshes_verts.norm(dim=2) # [M, N]
        normalized_error = torch.mean(per_vertex_errors.norm(dim=1) / gt_meshes_verts_norm.norm(dim=1), 0)
        self.surface_errors['3d_error'].append(normalized_error)
        self.log.write('Normalized vertex error(e_3D): {:05.6f}\n'.format(normalized_error))           
        return per_vertex_errors.detach().cpu()
    
    def compute_angular_error(self, recon_meshes, eps=1e-4): 
        """Computes angular error (eq. 2 in supplementary) between ground-truth meshes and reconstructed meshes
        Args:
            recon_meshes: Reconstucted meshes after aligning to ground-truth
        Returns:
        """
        recon_meshes_verts_normals = recon_meshes.verts_normals_padded()
        angle_error_cos = torch.cosine_similarity(recon_meshes_verts_normals, self.gt_meshes_verts_normals, dim=2)
        angle_error_radians = torch.where(angle_error_cos <= 1-eps, torch.acos(angle_error_cos), torch.tensor([0.], device=self.device)) 
        angle_error_degree_mean = torch.mean(180 * angle_error_radians / math.pi)
        self.surface_errors['angular_error'].append(angle_error_degree_mean)
        self.log.write('Normalized angular error(e_n): {:05.6f}\n'.format(angle_error_degree_mean)) 

    def plot_errors(self, save_dir):
        plots_dir = os.path.join(save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        fig = plt.figure(figsize=(13, 5))
        ax = fig.gca()
        for param, error in {**self.surface_errors}.items():
            ax.plot(np.arange(len(error)) * self.i_save, error)
            ax.set_xlabel("iteration", fontsize="16")
            ax.set_ylabel(param, fontsize="16")
            ax.set_title(param + ' vs iterations', fontsize="16")
            plt.savefig(os.path.join(plots_dir, '{}.png'.format(param)))
            plt.cla()
        torch.save(self.surface_errors, os.path.join(save_dir, 'surface_errors.pt'))