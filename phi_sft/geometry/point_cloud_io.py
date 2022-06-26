import os
import torch 
from pytorch3d.structures import Pointclouds

from geometry.obj_io import obj_read

def load_pointclouds_from_dir(pointclouds_dir, n_frames, device):
    pointcloud_files = [os.path.join(pointclouds_dir, f) for f in sorted(os.listdir(pointclouds_dir)) if f.endswith('.obj')][:n_frames]
    pointcloud_points_list = []
    point_clouds_lengths = []
    for pointcloud_file in pointcloud_files:
        pointcloud_points = obj_read(pointcloud_file)[0]
        point_clouds_lengths.append(pointcloud_points.shape[0])
        pointcloud_points_list.append(torch.Tensor(pointcloud_points).to(device))
    point_clouds = Pointclouds(points=pointcloud_points_list)
    point_clouds_lengths = torch.LongTensor(point_clouds_lengths).to(device)
    return point_clouds, point_clouds_lengths
