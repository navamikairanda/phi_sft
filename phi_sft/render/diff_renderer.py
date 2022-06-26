import os
import torch

import pdb
import numpy as np
from matplotlib import cm
from PIL import Image

# Data structures and functions for rendering
from pytorch3d.renderer import (
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    BlendParams
)
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    PointLights
)
from pytorch3d.transforms import axis_angle_to_matrix, axis_angle_to_matrix, quaternion_to_axis_angle, matrix_to_quaternion

from render.camera import get_kinect_camera, get_synthetic_camera
from image.image_io import save_images, save_images_np
from utils.json_io import read_json
from image.blur import get_gauss_instance
from geometry.mesh_io import load_meshes_from_template, load_meshes_from_dir
from render.depth_renderer import MeshRendererWithDepth

light_location = {'S1':[0.0, 0.0, 0.6], 'S2':[0.0, 0.0, 0.6], 'S3':[0.0, 0.0, 1.5], 
                  'S4':[0.0, 0.0, 0.6], 'S5':[0.0, 0.0, 0.6], 'S6':[0.0, 0.0, 1.8], 
                  'S7':[0.0, 0.0, 1.8], 'S8':[0.0, 0.0, 1.8], 'S9':[0.0, 0.0, 1.8]}
camera_novel_view = {'rotation': 
                { 'S1': [0, 2.1, 1.5], 'S2': [0, 2.1, 1.5], 'S3': [3.3, 2.1, 2.8], 
                  'S4': [0, 2.1, 1.8], 'S5': [0, 2.1, 1.5], 'S8': [0, 2.1, 1.5], 
                  'S9': [0, 2.1, 1.5], 'S7': [0, 2.1, 1.5], 'S6': [0, 2.1, 1.5]}, 
              'translation': 
                { 'S1': [[0.1, -0.5, 0.5]], 'S2': [[0.1, -0.5, 0.5]], 'S3': [[0.2, -0.1, 0.5]], 
                  'S4': [[0.2, -0.3, 0.5]], 'S5': [[0.1, -0.5, 0.5]], 'S8': [[0.1, -0.5, 0.5]],
                  'S9': [[0.1, -0.5, 0.5]], 'S7': [[0.1, -0.5, 0.5]], 'S6': [[0.1, -0.5, 0.5]]}
            }

class DifferentiableRenderer():
    def __init__(self, args_expt, args_data, device, evaluation=False):
        self.device = device
        self.sequence_type = args_data['sequence_type']
        self.sequence_name = args_data['sequence_name']
        self.tex_image = torch.from_numpy(np.asarray(Image.open(args_data['texture_file']), dtype=np.float32)[...,:3]/ 255.).unsqueeze(0).to(self.device) 
        self.template_file = args_data['template_file']       
        self.n_render = args_expt.getint('n_render')

        calibration = read_json(os.path.join(args_data['sequence_dir'], "camera.json"))
        if self.sequence_type == 'real':
            self.min_depth = args_data.getfloat('min_z')
            self.max_depth = args_data.getfloat('max_z')
            self.cameras, self.image_size = get_kinect_camera(calibration, self.device)
        elif self.sequence_type == 'synthetic':
            self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
            self.cameras, self.image_size = get_synthetic_camera(calibration, self.device) 
        
        if evaluation: 
            self.lights = PointLights(device=self.device, location=[light_location[args_data['sequence_name']]], ambient_color=((0.3, 0.3, 0.3), ), diffuse_color=((0.5, 0.5, 0.5), ), specular_color=((0.2, 0.2, 0.2),))
            background_color = (1.0, 1.0, 1.0)
        else: #Optimisation
            if self.sequence_type == 'real':
                self.lights = AmbientLights(device=self.device)
                background_color = (0.0, 0.0, 0.0)
            elif self.sequence_type == 'synthetic':
                self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
                background_color = (1.0, 1.0, 1.0)
        self.raster_settings_hard = RasterizationSettings( 
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        sigma = 1e-6
        self.raster_settings_soft = RasterizationSettings( 
            image_size=self.image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, #0.000921024036697585
            faces_per_pixel=args_expt.getint('faces_per_pixel'), 
        )      
        sigma = 1e-4
        gamma = 1e-4
       
        blend_params = BlendParams(sigma, gamma, background_color)

        self.shader_soft = SoftPhongShader(
            device=self.device, 
            lights=self.lights,
            cameras=self.cameras, 
            blend_params=blend_params
            )
        self.shader_hard = HardPhongShader(
            device=self.device, 
            lights=self.lights,
            cameras=self.cameras, 
            blend_params=blend_params
            )
        self.gauss = get_gauss_instance(args_data)

    def _render_rgba_vis(self, meshes, blur, novel_view): 
        """
		Renders the reconstructed surfaces from input or novel view for visualisation
		Args:
			meshes: Meshes to render
			blur: True if the rendered masks images should be blurred
            novel_view: True if to render meshes from novel camera view point
		Returns:
			images: Rendered RGB images.
            depths: Rendered depth images. 
		"""
        if novel_view:
            axis_angle = quaternion_to_axis_angle(matrix_to_quaternion(self.cameras.R))
            axis_angle += torch.tensor(camera_novel_view['rotation'][self.sequence_name], device=self.device)
            matrix = axis_angle_to_matrix(axis_angle)
            self.cameras.R = matrix
            self.cameras.T += torch.tensor(camera_novel_view['translation'][self.sequence_name], device=self.device)

        renderer = MeshRendererWithDepth(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings_hard
            ),
            shader=self.shader_hard
        )
        
        images = []
        depths = []
        for i in range(0, len(meshes), self.n_render):  
            image, depth = renderer(meshes[i:i+ self.n_render])
            images.append(image)
            depths.append(depth)
        images = torch.cat(images, dim=0)
        ## Gaussian blur of alphas
        if blur:
            alphas = images[...,3].unsqueeze(dim=3)
            alphas = torch.transpose(alphas, 3, 1)
            alphas = self.gauss(alphas)
            
            alphas = torch.transpose(alphas, 3, 1)
            images = torch.cat((images[...,:3], alphas), dim=3)
        depths = torch.cat(depths, dim=0)
        return images, depths
    
    def _render_rgba_optim(self, meshes): 
        """
		Renders the reconstructed surfaces from input view for optimisation
		Args:
			meshes: Meshes to render
		Returns:
			images: Rendered RGBA images.
		"""
        rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=self.raster_settings_soft
            )
        soft_renderer=MeshRenderer(
            rasterizer=rasterizer,
            shader=self.shader_soft
        )
        hard_renderer=MeshRenderer(
            rasterizer=rasterizer,
            shader=self.shader_hard
        )
        
        rgbs = []
        alphas = []
        for i in range(0, len(meshes), self.n_render):  
            rgbs.append(soft_renderer(meshes[i:i+ self.n_render])[...,:3])
            alphas.append(hard_renderer(meshes[i:i+ self.n_render])[...,3][...,None])
        rgbs = torch.cat(rgbs, dim=0)
        alphas = torch.cat(alphas, dim=0)
        ## Gaussian blur of alphas
        alphas = torch.transpose(alphas, 3, 1)
        alphas = self.gauss(alphas)
        alphas = torch.transpose(alphas, 3, 1)
        images = torch.cat((rgbs, alphas), dim=3)
        return images

    def render_pointcloud(self, pointclouds): 
        raster_settings = PointsRasterizationSettings(
            image_size = self.image_size, 
            radius = 0.01,
            points_per_pixel = 10, 
            bin_size = 0
        )
        rasterizer = PointsRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(0, 0, 0))
        )
        
        images = []
        for i in range(0, len(pointclouds), self.n_render):  
            image = renderer(pointclouds[i:i+ self.n_render])
            images.append(image)
        images = torch.cat(images, dim=0)
        return images

    def render_rgba_optim(self, meshes_verts):
        meshes = load_meshes_from_template(self.template_file, meshes_verts, tex_image=self.tex_image, device=self.device) 
        images = self._render_rgba_optim(meshes)
        return images

    def render_rgba_vis(self, base_dir, meshes_verts=None):
        """
		Renders reconstructed surfaces from input view and save the RGB images, masks and depths
		Args:
            base_dir: Path to meshes if rendering saved meshes
			meshes_verts: Mesh vertices to render if using topology from template
		Returns:
			None
		"""
        if meshes_verts is None:
            surfaces_dir = os.path.join(base_dir, 'surfaces')
            meshes = load_meshes_from_dir(surfaces_dir, tex_image=self.tex_image, device=self.device) 
        else:
            meshes = load_meshes_from_template(self.template_file, meshes_verts, tex_image=self.tex_image, device=self.device)
        images, depths = self._render_rgba_vis(meshes, True, False)  
        save_images(images[...,:3], os.path.join(base_dir, 'rgbs'))
        save_images(images[...,3], os.path.join(base_dir, 'blurred_masks'))
        if self.sequence_type == 'real':
            output_depths = (depths.squeeze() - self.min_depth) / (self.max_depth - self.min_depth)
            output_depths = cm.jet(output_depths.cpu().numpy())
            save_images_np(output_depths, os.path.join(base_dir, 'depths'))
    
    def render_mesh_vis(self, base_dir, meshes, novel_view):
        """
		Renders the reconstructed surfaces from input or novel view ans save the mesh images and depths - only used with evaluation script
		Args:
			meshes: Meshes to render
            novel_view: True if to render meshes from novel camera view point
		Returns:
			None 
		"""
        images, depths = self._render_rgba_vis(meshes, False, novel_view)  
        meshes_dir = os.path.join(base_dir, self.sequence_name + '_meshes_novel_view') if novel_view else os.path.join(base_dir, self.sequence_name + '_meshes')
        save_images(images[...,:3], meshes_dir)
        if self.sequence_type == 'real' and not novel_view:
            output_depths = (depths.squeeze() - self.min_depth) / (self.max_depth - self.min_depth)
            output_depths = cm.jet(output_depths.cpu().numpy())
            output_depths[~images[...,3].cpu().numpy().astype(np.bool)] = np.array([1., 1., 1., 0.])
            save_images_np(output_depths, os.path.join(base_dir, self.sequence_name + '_depths'))
              