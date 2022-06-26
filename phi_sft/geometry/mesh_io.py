import os
import natsort
import torch
import numpy as np
import pdb

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import TexturesUV, TexturesVertex

def load_meshes_from_dir(obj_dir, tex_image=None, white_verts_features=False, device=None, n_frames=None, re_orient_faces=False):
    """Load meshes from a directory containing .obj files using the load_obj function, and
    return them as a Meshes object. 
    Args:
        obj_dir: Path to meshes
        tex_image: UV texture for the meshes
        white_verts_features: [1, n_vertices, 3]. Vertex textures for the meshes
        n_frames: Vertex textures for the meshes
        re_orient_faces: Change orientation of faces from positive to negative or viceversa
    Returns:
        New Meshes object.
    """
    mesh_list = []
    obj_files = [os.path.join(obj_dir, f) for f in  natsort.natsorted(os.listdir(obj_dir)) if f.endswith('obj')][:n_frames]
    for i, f_obj in enumerate(obj_files):
        verts, faces, aux = load_obj(f_obj, device=device)
        if tex_image is not None:
            tex = TexturesUV(verts_uvs=[aux.verts_uvs], faces_uvs=[faces.textures_idx], maps=tex_image)
        elif white_verts_features:
            tex = TexturesVertex(verts_features=torch.ones_like(verts)[None, ...])
        else: 
            tex = None
        faces_verts_idx = faces.verts_idx
        if re_orient_faces:
            faces_verts_idx = torch.stack((faces_verts_idx[...,2],faces_verts_idx[...,1],faces_verts_idx[...,0]),1)
        mesh = Meshes(verts=[verts], faces=[faces_verts_idx], textures=tex)
        mesh_list.append(mesh)
    if len(mesh_list) == 1:
        return mesh_list[0]
    return join_meshes_as_batch(mesh_list)

def load_meshes_from_template(template_obj_file, meshes_verts, tex_image=None, verts_features=None, device=None):
    """
    Load meshes with the topology of the given 3D template and the vertex locations for all surfaces and  
    return them as a Meshes object. 

    Args:
        template_obj_file: Path to 3D template
        meshes_verts: [n_frames, n_vertices, 3] Vertex positions for all surfaces
        tex_image: UV texture for the meshes

    Returns:
        New Meshes object.
    """
    _, faces, aux = load_obj(template_obj_file, device=device)
    n_meshes = meshes_verts.shape[0]
    if tex_image is not None:
        textures = TexturesUV(verts_uvs=aux.verts_uvs.repeat(n_meshes, 1, 1), faces_uvs=faces.textures_idx.repeat(n_meshes, 1, 1), maps= tex_image.repeat(n_meshes, 1, 1, 1))
    elif verts_features is not None:
        pdb.set_trace() #TODO not verified
        textures = TexturesVertex(verts_features=torch.from_numpy(verts_features[np.newaxis,0,:,:3]).to(device))
    else: 
        textures = None
    meshes = Meshes(meshes_verts.float(), faces.verts_idx.repeat(n_meshes, 1, 1),  textures=textures)
    return meshes

def save_meshes_to_dir(obj_dir, template_obj_file, meshes_verts, device=None):
    """
    Save meshes with the topology of the given 3D template and the vertex locations for all surfaces

    Args:
        template_obj_file: Path to 3D template
        meshes_verts: [n_frames, n_vertices, 3] Vertex positions for all surfaces
    Returns:
        
    """
    os.makedirs(obj_dir, exist_ok=True)
    _, faces, _ = load_obj(template_obj_file, device=device)
    for mesh_idx, mesh in enumerate(meshes_verts):
        save_obj(os.path.join(obj_dir, '%04d_00.obj'%(mesh_idx)), mesh, faces.verts_idx)
