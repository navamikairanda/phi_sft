import os
import torch
import matplotlib.pyplot as plt

from image.image_io import read_images, save_images

class ImageLoss():
	
	def __init__(self, diff_renderer, args_expt, args_data, log, device):
		self.diff_renderer = diff_renderer
		self.log = log
		self.args_expt = args_expt
		if args_expt.getboolean('reload'): 
			self.losses = torch.load(os.path.join(self.args_expt['log_dir'], args_expt['i_reload'], 'losses.pt'))
		else: 
			self.losses = {loss:[] for loss in ['rgb', 'silhouette']} 
		self.target_rgbs = read_images(os.path.join(args_data['sequence_dir'], 'rgbs'), n_imgs = args_expt.getint('n_frames'), device=device)
		self.target_silhouettes = read_images(os.path.join(args_data['sequence_dir'], 'blurred_masks'), n_imgs = args_expt.getint('n_frames'), device=device)
		
	def vis_loss(self, save_dir, recon_meshes_verts):
		"""
		Renders the reconstructed surfaces, visualise and save image loss 
		Args:
			save_dir: 
			recon_meshes_verts: Vertex positions of reconstructed surfaces.
		Returns:
			None
		"""
		predicted_images = self.diff_renderer.render_rgba_optim(recon_meshes_verts)
		mse_rgbs = (predicted_images[..., :3] - self.target_rgbs) ** 2
		mse_silhouette = ((predicted_images[...,3] - self.target_silhouettes) ** 2)
		save_images(mse_rgbs, os.path.join(save_dir, 'rgb_loss'))
		save_images(mse_silhouette, os.path.join(save_dir, 'silhouette_loss'))

	def compute_loss(self, recon_meshes_verts):
		"""
		Compute photometric and silhouette losses
		Args:
			recon_meshes_verts: Vertex positions of reconstructed surfaces. 
		Returns:
			loss: Sum of photometric and silhouette loss 
			loss_last_frame: Loss for the last frame in the sequence. This is used as a parameter in determining adapative optimisation flow.
		"""
		predicted_image = self.diff_renderer.render_rgba_optim(recon_meshes_verts)
		# Normalize the RGB values to remove intensity and retain only color
		if(self.args_expt['loss'] == 'L2'):
			loss_rgb = ((predicted_image[..., :3] - self.target_rgbs[:predicted_image.shape[0]]) ** 2).mean()
			loss_silhouette = ((predicted_image[...,3] - self.target_silhouettes[:predicted_image.shape[0]]) ** 2).mean() 
		elif(self.args_expt['loss'] == 'L1'):
			loss_rgb = torch.abs(self.target_rgbs[:predicted_image.shape[0]] - predicted_image[..., :3]).mean()
			loss_silhouette = torch.abs(predicted_image[...,3] - self.target_silhouettes[:predicted_image.shape[0]]).mean() 
			loss_last_frame = torch.abs(self.target_rgbs[predicted_image.shape[0]-1] - predicted_image[..., :3][-1]).mean().detach()
		loss = self.args_expt.getfloat('w_rgb') * loss_rgb + self.args_expt.getfloat('w_sil') * loss_silhouette

		self.losses['rgb'].append(self.args_expt.getfloat('w_rgb') * loss_rgb.detach())
		self.losses['silhouette'].append(self.args_expt.getfloat('w_sil') * loss_silhouette.detach())
		return loss, loss_last_frame

	def plot_loss(self, save_dir):
		torch.save(self.losses, os.path.join(save_dir, 'losses.pt'))
		plots_dir = os.path.join(save_dir, 'plots')
		os.makedirs(plots_dir, exist_ok=True)
		fig = plt.figure(figsize=(13, 5))
		ax = fig.gca()
		for loss_type, loss_values in self.losses.items():
			ax.plot(loss_values)
			ax.set_xlabel("iteration", fontsize="16")
			ax.set_ylabel(loss_type + ' loss', fontsize="16")
			ax.set_title(loss_type + ' loss vs iterations', fontsize="16")
			plt.savefig(os.path.join(plots_dir, '{}.png'.format(loss_type)))
			plt.cla()
