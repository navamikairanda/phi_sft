import torch
import os
import pdb

from pytorch3d.ops import iterative_closest_point, corresponding_points_alignment
from geometry.point_cloud_io import load_pointclouds_from_dir
from geometry.mesh_io import load_meshes_from_dir

class Align():

    def __init__(self, args_expt, args_data, device):
        self.device = device
        self.sequence_type = args_data['sequence_type']
        self.re_orient_faces = args_expt.getboolean('re_orient_faces')
        self.per_frame_registration = args_expt.getboolean('per_frame_registration')

        if self.sequence_type == 'real':
            self.gt_point_clouds, self.gt_point_clouds_lengths = load_pointclouds_from_dir(os.path.join(args_data['sequence_dir'], 'point_clouds'), args_expt.getint('n_frames'), device)
        elif self.sequence_type == 'synthetic':
            gt_meshes = load_meshes_from_dir(os.path.join(args_data['sequence_dir'], 'surfaces'), device=self.device)[:args_expt.getint('n_frames')]
            self.gt_meshes_verts = gt_meshes.verts_padded()
            
    def align_recon_2_gt(self, recon_dir, white_verts_features=False):
        """Aligns reconstructed meshes to groud-truth meshes/point clouds using Procrustes alignment for synthetic sequences (point correspondence is available) and Iterative Closest Point method for real sequences (point correspondence is not available))
        Args:
            recon_dir: path to reconstructed meshes (obj files)
            white_verts_features: vertex textures is white color
        Returns:
            aligned_meshes: meshes after aligning to ground-truth
        """
        recon_meshes = load_meshes_from_dir(recon_dir, white_verts_features=white_verts_features, re_orient_faces=self.re_orient_faces, device=self.device)
        if self.sequence_type == 'real':
            if self.per_frame_registration:
                recon_2_gt_icp = iterative_closest_point(recon_meshes.verts_padded(), self.gt_point_clouds)
                aligned_meshes = recon_meshes.update_padded(recon_2_gt_icp.Xt)
            else: 
                aligned_meshes = recon_meshes
        elif self.sequence_type == 'synthetic':
            recon_2_gt_procrsustes = corresponding_points_alignment(recon_meshes.verts_padded(), self.gt_meshes_verts, estimate_scale=True)        
            recon_meshes_verts_aligned = recon_2_gt_procrsustes.s[:, None,None] * torch.bmm(recon_meshes.verts_padded(), recon_2_gt_procrsustes.R) + recon_2_gt_procrsustes.T[:, None, :]
            aligned_meshes = recon_meshes.update_padded(recon_meshes_verts_aligned) 
        return aligned_meshes