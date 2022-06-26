import torch
import pdb
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform,
    FoVPerspectiveCameras
)

def get_kinect_camera(calibration, device):
    """
    Camera specifications for real sequences recorded with Azure Kinect
    https://docs.microsoft.com/en-us/azure/kinect-dk/coordinate-systems
    Args:
        calibration: Camera intrinsics
    Returns:
        cameras: Perspecitve camera
        image_size: [H, W]
    """
    R = torch.tensor([[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]]) # Extrinsics rotation matrix R holds the coordinate transform from Kinect SDK to PyTorch3D conventions
    image_size = (calibration['height'], calibration['width'])
    focal_length = (calibration['fx'], calibration['fy'])
    principal_point = (calibration['cx'], calibration['cy'])

    cameras = PerspectiveCameras(focal_length=(focal_length,), principal_point=(principal_point,), in_ndc=False, image_size=(image_size,), R=R, device=device)
    return cameras, image_size

def get_synthetic_camera(calibration, device):
    """
    Camera specifications for synthetic sequences generated with physics simulator
    Args:
        calibration: Camera calibration
    Returns:
        cameras: Field of View Perspecitve camera
        image_size: [H, W]
    """
    object_pos = torch.tensor(calibration["object_pos"], device=device)[None, :]
    # camera view 1, camera to object center distance --> 1.8m
    x_dir = ((0, 0, 1),)
    camera_pos = torch.tensor(calibration["camera_pos"], device=device)[None, :]
    R, T = look_at_view_transform(eye=camera_pos, up=x_dir, at=object_pos) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    image_size = (calibration['height'], calibration['width'])
    return cameras, image_size
