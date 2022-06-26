import os
import numpy as np
import pymeshlab
import math
import torch
import torch.nn.functional as F
import imageio
import pdb
import time

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.ops import iterative_closest_point, sample_points_from_meshes
from pytorch3d.ops.points_alignment import SimilarityTransform
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle

from utils.json_io import read_json, save_json
from render.camera import get_kinect_camera

def _poisson_reconstruction(template_pointcloud_name, template_mesh_name):
    """
	Poisson surface reconstruction
	Args:
		template_pointcloud_name: Input point cloud for 3D template
		template_mesh_name: Reconstructed mesh
	Returns:
		None
	"""
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(template_pointcloud_name)
    ms.compute_normals_for_point_sets()
    ms.surface_reconstruction_screened_poisson(depth=4, scale=1.0, fulldepth=3)
    ms.save_current_mesh(template_mesh_name)

def _generate_texture_map(template_mesh_name, textured_mesh_name, texture_file_name, sequence_dir, device):
    """
	Generate texture map for 3D template using RGB image of first frame. Texture coordinates are projections of mesh vertices with given camera

	Args:
		template_mesh_name: 3D template
        textured_mesh_name: Textured 3D template
        texture_file_name: Texture map image
		
	Returns:
		None
	"""
    verts, faces, _ = load_obj(template_mesh_name, device=device)
    verts = verts[None,...]

    calibration = read_json(os.path.join(sequence_dir, 'camera.json'))
    cameras, image_size = get_kinect_camera(calibration, device)
    # Project mesh vertices to generate UV coordinates
    verts_uvs = cameras.transform_points_screen(verts)[:, :, :2]
    image_size_tensor = torch.tensor([image_size[1],image_size[0]], device=device)
    # Screen space to normalized device coordinates
    verts_uvs = torch.div(verts_uvs, image_size_tensor)

    # Prepare texture image
    texture_image = imageio.imread(texture_file_name)/255.
    texture_image = np.flipud(texture_image).copy()
    texture_image = torch.from_numpy(texture_image).to(device)
    
    save_obj(textured_mesh_name, verts.squeeze(), faces.verts_idx, verts_uvs=verts_uvs.squeeze(), faces_uvs=faces.verts_idx, texture_map=texture_image.squeeze())

def generate_template_surface(sequence_dir, device):
    template_dir = os.path.join(sequence_dir, 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    template_pointcloud_name = os.path.join(sequence_dir, 'point_clouds', 'point_cloud_000.obj')
    template_mesh_name = os.path.join(template_dir, 'template_mesh_init.obj')
    _poisson_reconstruction(template_pointcloud_name, template_mesh_name)

    texture_file_name = os.path.join(sequence_dir, 'rgbs', 'rgb_000.png') 
    textured_mesh_name = os.path.join(template_dir, 'template_mesh_init_textured.obj')
    _generate_texture_map(template_mesh_name, textured_mesh_name, texture_file_name, sequence_dir, device)

def _create_simulator_template(template_mesh_name, simulator_template_mesh_name, simulator_config_name, args_data, device):
    """
	Given mesh generated from Kinect RGBD for first frame, prepare the template for physics siumulator. Simulator requires that the template should be in meters as the unit and always in XY plane, and the template location, and initial rigid pose should be added in the simulator configuration file (sim_conf.json)

	Args:
		template_mesh_name: 3D template generated previously as first (template) frame RGBD --> point cloud --> mesh 
        simulator_template_mesh_name: Template mesh in simulator expected format
        simulator_config_name: Simulator configuration updated with template details (path to template and pose of the template in the world coordinate system)
		
	Returns:
		None
		
	"""
    # src is kinect template, trg is simulator template
    src_verts, src_faces, _ = load_obj(template_mesh_name, device=device)
    src_mesh = Meshes(verts=[src_verts], faces=[src_faces.verts_idx])
    src_point_cloud = Pointclouds(points=sample_points_from_meshes(src_mesh, args_data.getint('n_points_sample')))

    # Sample pointcloud in xy-plane with (x,y,z) ~ [-cloth_size/2, cloth_size/2]^2, 0 grid 
    N = args_data.getint('trg_vert_step')
    edge_len_x = args_data.getfloat('edge_len_x') #0.55 for 55cmx55cm cloth
    edge_len_y = args_data.getfloat('edge_len_y')
    trg_verts_x, trg_verts_y = torch.meshgrid(torch.linspace(-edge_len_x/2, edge_len_y/2, N), torch.linspace(-edge_len_x/2, edge_len_y/2, N))
    trg_verts = torch.stack([trg_verts_x, trg_verts_y, torch.zeros_like(trg_verts_y)], -1).reshape((-1, 3))
    trg_point_cloud = Pointclouds(points=[trg_verts.to(device)])
    
    # Initialize the pose by shifting centroid of cloth to (0, 0, 0)
    R = torch.eye(3, device=device).unsqueeze(0)
    T = -1 * torch.mean(src_verts, 0).unsqueeze(0)
    s = torch.tensor([1], device=device)
    init_transform = SimilarityTransform(R, T, s)

    src_2_trg_pose = iterative_closest_point(src_point_cloud, trg_point_cloud, init_transform=init_transform) #allow_reflection=True
    T = src_2_trg_pose.RTs.T.squeeze()
    R_matrix = src_2_trg_pose.RTs.R.squeeze()

    # Simulator requires [angle, x, y, z] format
    # Invert the matrix to find pose from simulator template to kinect template
    R_axis_angle = quaternion_to_axis_angle(matrix_to_quaternion(R_matrix))
    R_axis_angle_sim = F.normalize(R_axis_angle, dim=0, p=2).tolist()
    R_angle_degrees = R_axis_angle.norm().item() * 180 / math.pi
    R_axis_angle_sim.insert(0, R_angle_degrees)

    sim_config = read_json(simulator_config_name)
    sim_config['cloths'][0]['mesh'] = simulator_template_mesh_name
    sim_config['cloths'][0]['transform']['rotate'] = R_axis_angle_sim
    sim_config['cloths'][0]['transform']['translate'] = (-torch.matmul(T, R_matrix.inverse())).tolist()
    save_json(sim_config, simulator_config_name)

    mesh_unposed_verts = torch.mm(src_verts, R_matrix) + T
    save_obj(simulator_template_mesh_name, mesh_unposed_verts, src_faces.verts_idx)

def _invert_faces_orientation(template_mesh_name):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(template_mesh_name)
    ms.invert_faces_orientation()
    ms.save_current_mesh(template_mesh_name)
    time.sleep(10)

def clean_template_surface(sequence_dir, args_data, device):
    template_dir = os.path.join(sequence_dir, 'templates')   
    template_mesh_name = os.path.join(template_dir, 'template_mesh_final.obj')

    texture_file_name = os.path.join(sequence_dir, 'rgbs', 'rgb_000.png')  
    textured_mesh_name = os.path.join(template_dir, 'template_mesh_final_textured.obj')
    _generate_texture_map(template_mesh_name, textured_mesh_name, texture_file_name, sequence_dir, device)
    if args_data.getboolean('invert_faces_orientation'):
        _invert_faces_orientation(template_mesh_name)

    simulator_template_mesh_name = os.path.join(template_dir, 'template_mesh_final_untextured_unposed.obj')
    simulator_config_name = os.path.join(sequence_dir, 'sim_conf.json')
    _create_simulator_template(template_mesh_name, simulator_template_mesh_name, simulator_config_name, args_data, device)
