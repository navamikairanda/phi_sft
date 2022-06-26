import os
import torch
import imageio
import numpy as np

to8b = lambda x : (255 * np.clip(x, 0, 1)).astype(np.uint8)
def read_images_np(img_dir, start_frame_idx=1, n_imgs=None):
	img_files = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir)) if f.endswith('png')]
	img_files = img_files[start_frame_idx-1:][:n_imgs]
	images = [imageio.imread(f)/255. for f in img_files]
	return images

def read_images(img_dir, start_frame_idx=1, n_imgs=None, device=torch.device("cuda:0")):
	images_np = read_images_np(img_dir, start_frame_idx, n_imgs)
	images = torch.tensor(images_np, device=device)
	return images

def save_images_np(images, img_dir, save_video=True, image_prefix='image_', start_frame_idx=0):
	os.makedirs(img_dir, exist_ok=True)
	for i, f in enumerate(images):
		imageio.imwrite(os.path.join(img_dir, image_prefix + '{:03d}.png'.format(start_frame_idx + i)), to8b(f)) 
	if save_video:
		imageio.mimwrite(os.path.join(img_dir, os.path.basename(img_dir) + '.mp4'), to8b(images), fps=5, quality=8, macro_block_size = 8)

def save_images(images, img_dir, save_video=True, image_prefix='', start_frame_idx=0):
	save_images_np(images.detach().cpu().numpy(), img_dir, save_video, image_prefix, start_frame_idx)
