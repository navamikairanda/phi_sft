import torch
import sys
import os
import pdb
from tqdm import tqdm
from shutil import copyfile

from geometry.mesh_io import save_meshes_to_dir
from image.image_loss import ImageLoss
from render.diff_renderer import DifferentiableRenderer
from simulation.physics_params_helper import init_physics_params, update_material_reuse
from simulation.physics_simulator import PhysicsSimulator
from utils.device import get_device
from utils.configparser import config_parser
from evaluation.metrics import Metrics
from evaluation.align import Align

'''
sequence_type=real
sequence_name=s3
cd ${code_root}/phi_sft
python -u reconstruct_surfaces_from_sequence.py ${data_root}/${sequence_type} ${code_root}/phi_sft/config/expt_${sequence_type}_${sequence_name}.ini
'''

def train(args_expt, args_data):
	"""
	Monocular non-rigid 3D reconstruction / Shape-from-Template with a physics-based deformation model
	"""
	device = get_device()
	log_file = os.path.join(args_expt['log_dir'], 'log.txt')

	if args_expt.getboolean('reload'):
		log = open(log_file, 'a+', buffering=1)
		log.write('\nResuming experiment {} from iter {}\n'.format(args_expt['expt_name'], args_expt['i_reload']))

		ckpt_path = os.path.join(args_expt['log_dir'], args_expt['i_reload'], 'ckpt_{:04d}.tar'.format(args_expt.getint('i_reload')))
		log.write('\nReloading from {}\n'.format(ckpt_path))
		ckpt = torch.load(ckpt_path)
		
		model_params = ckpt['model_params']
		loss_reference_frame = ckpt['loss_reference_frame']
		update_material_reuse(args_expt['log_dir'], False)
		start = args_expt.getint('i_reload') + 1
	else: 
		log = open(log_file, 'w', buffering=1)
		log.write('Running experiment {} \n'.format(args_expt['expt_name']))
		
		copyfile(os.path.join(args_data['sequence_dir'], 'sim_conf.json'), os.path.join(args_expt['log_dir'], 'sim_conf.json'))
		ckpt = None
		model_params = init_physics_params(args_expt, args_data, log)
		loss_reference_frame = 100 #TODO set to max
		start = 1
	
	diff_renderer = DifferentiableRenderer(args_expt, args_data, device)
	metrics = Metrics(args_expt, args_data, log, device)
	align = Align(args_expt, args_data, device) 
	image_loss_object = ImageLoss(diff_renderer, args_expt, args_data, log, device)
	physics_sim = PhysicsSimulator(args_expt['log_dir'], device)

	sim_grad_vars = []
	for key in model_params.keys():
		sim_grad_var = {'params': model_params[key], 'lr': float(args_expt[key])}
		sim_grad_vars.append(sim_grad_var)	
	optimizer = torch.optim.Adam(sim_grad_vars)

	if ckpt != None:  
		optimizer.load_state_dict(ckpt['optimizer_state_dict'])
	
	n_epoch = 0
	iter_per_new_frame = 0
	max_frame_idx = args_expt.getint('reference_frame_for_loss')
	for cur_iter in tqdm(range(start, args_expt.getint('max_iterations'))): 
		if cur_iter > start and ((loss_cur_frame <= loss_reference_frame + 0.00001) or iter_per_new_frame >= args_expt.getint('max_iterations_per_new_frame')):
			log.write('Spent {} iterations with {} frames!\n'.format(iter_per_new_frame, max_frame_idx))
			max_frame_idx = min(max_frame_idx + 1, args_expt.getint('n_frames'))
			iter_per_new_frame = 0
		if max_frame_idx == args_expt.getint('n_frames'):
			n_epoch += 1
			log.write('Epoch {} complete!\n'.format(n_epoch))

		if cur_iter % (args_expt.getint('i_save')) == 0 or cur_iter == 1:
			with torch.no_grad():
				physics_sim.reset(model_params)
				recon_meshes_verts = physics_sim.run(args_expt.getint('n_frames'), model_params['correctives'])
				save_dir = os.path.join(args_expt['log_dir'], str(cur_iter))
				save_meshes_to_dir(os.path.join(save_dir, 'surfaces'), diff_renderer.template_file, recon_meshes_verts, device)
				# TODO load meshes here, instead of separately in diff_renderer and error compute metric
				diff_renderer.render_rgba_vis(save_dir, recon_meshes_verts)
				aligned_meshes = align.align_recon_2_gt(os.path.join(save_dir, 'surfaces'))
				metrics.compute_recon_errors(aligned_meshes)
				image_loss_object.vis_loss(save_dir, recon_meshes_verts)			
				image_loss_object.plot_loss(save_dir)
				metrics.plot_errors(save_dir)
		
			ckpt_path = os.path.join(args_expt['log_dir'], str(cur_iter), 'ckpt_{:04d}.tar'.format(cur_iter))
			torch.save({
				'model_params': model_params,
				'optimizer_state_dict': optimizer.state_dict(),
				'loss_reference_frame': loss_reference_frame,
			}, ckpt_path)
			log.write('\nSaved checkpoints at {}\n'.format(ckpt_path))

		if cur_iter == start + 1:
			update_material_reuse(args_expt['log_dir'], True)
		optimizer.zero_grad()
		physics_sim.reset(model_params)
		sim_meshes_verts = physics_sim.run(max_frame_idx,  model_params['correctives'])
		loss, loss_cur_frame = image_loss_object.compute_loss(sim_meshes_verts)
		if cur_iter == 1: 	
			loss_reference_frame = loss_cur_frame
			log.write('RGB loss with {} frames is {}\n'.format(args_expt.getint('reference_frame_for_loss'), loss_reference_frame))
		iter_per_new_frame += 1
		loss.backward()
		optimizer.step()					

if __name__=='__main__':
	data_dir = sys.argv[1] 
	args_expt_file = sys.argv[2]
	args_expt = config_parser(args_expt_file)['DEFAULT'] 
    
	args_data = config_parser(os.path.join(data_dir, args_expt['sequence_name'], 'preprocess.ini'))['DEFAULT']
	
	os.makedirs(args_expt['log_dir'], exist_ok=True)
	copyfile(args_expt_file, os.path.join(args_expt['log_dir'], 'args.ini'))

	train(args_expt, args_data)
