import os
import torch
import pdb

from utils.json_io import read_json, save_json
from pytorch3d.io import load_obj

def init_physics_params(args_expt, args_data, log):
	"""
	Initialize physical parameters to reasonable values. External forces such as wind, gravity and correctives are set to zero. Surface material elastic properties are initialized to average values over 10 different materials desribed in Wang et al.. 
    A subset of physics parameters can be optimized for synthetic sequences while initializing other parameters to known values.

	Args:
		args_expt: Experiment arguments
		args_data: Data sequence arguments
		
	Returns:
		model_params: List of physical parameters and their initial values. 
		
	"""
	config = read_json(os.path.join(args_expt['log_dir'], 'sim_conf.json'))
	
	# Corrective forces, [n_frames, n_verts, 3]
	n_verts = load_obj(args_data['template_file'])[0].shape[0]
	correctives = torch.zeros([args_expt.getint('n_frames'), n_verts, 3], dtype=torch.float64, requires_grad=args_expt.getboolean('opt_correctives'))
	
	config['gravity'] = [0, 0, 0]

	if args_expt.getboolean('opt_wind'):
		wind_velocity = torch.tensor([0, 0, 0], dtype=torch.float64, requires_grad=True)
		config['wind']['velocity'] = wind_velocity.detach().numpy().tolist()
	else:
		wind_velocity = torch.tensor(config['wind']['velocity'], dtype=torch.float64, requires_grad=False)

	if args_expt.getboolean('opt_material'):
		material = read_json(os.path.join(args_data['data_dir'], 'materials', 'init_mat.json')) 
		density = torch.tensor(material['density'], dtype=torch.float64, requires_grad=True)
		stretch = torch.tensor(material['stretching'], dtype=torch.float64, requires_grad=True)
		bend = torch.tensor(material ['bending'], dtype=torch.float64, requires_grad=True)
		config['cloths'][0]['materials'][0]['data'] = os.path.join(args_data['data_dir'], 'materials', 'init_mat.json')
		config['cloths'][0]['materials'][0]['stretching_mult'] = 1
		config['cloths'][0]['materials'][0]['bending_mult'] = 1
	else:
		material = read_json(config['cloths'][0]['materials'][0]['data']) 
		density = torch.tensor(material['density'], dtype=torch.float64, requires_grad=False)
		stretch = torch.tensor(material['stretching'], dtype=torch.float64, requires_grad=False)
		bend = torch.tensor(material ['bending'], dtype=torch.float64, requires_grad=False)
	
	if not args_expt.getboolean('is_handles_known'):
		config['handles'][0]['nodes'] = []

	model_params = {'correctives': correctives, 
			'wind_velocity': wind_velocity, 
			'density': density, 
			'stretch': stretch, 
			'bend': bend } 
	log.write('\nInitial values for physical parameters \n')
	for key, value in model_params.items():
		log.write('{} = {} \n'.format(key, value))
	save_json(config, os.path.join(args_expt['log_dir'], 'sim_conf.json'))
	return model_params

def update_material_reuse(sim_conf_dir, reuse):
	"""
	Material re-use should be set to False when starting or reloading physics simulation in surface reconstruction. After first simulation run, re-use needs to be set to True

	Args:
		sim_conf_dir: Path to simulation configuration
		reuse: True if loaded material should be re-used, else False
		
	Returns:
		None	
	"""
	config = read_json(os.path.join(sim_conf_dir, 'sim_conf.json'))
	config['cloths'][0]['materials'][0]['reuse'] = reuse 
	save_json(config, os.path.join(sim_conf_dir, 'sim_conf.json'))