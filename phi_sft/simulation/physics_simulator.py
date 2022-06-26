import arcsim
import os
import sys
import torch
import pdb
from utils.json_io import read_json
sys.path.append('../pysim')

class PhysicsSimulator():

	def __init__(self, sim_dir, device):
		torch.set_num_threads(8)
		self.device = device
		self.sim_dir = sim_dir
		self.frame_steps = read_json(os.path.join(self.sim_dir, 'sim_conf.json'))['frame_steps']
		self.sim = arcsim.get_sim()

	def reset(self, model_params=None): 
		"""
		Reset simulator by initialising physics (such as template, material and forces) from simulator configuration file (sim_conf.json). Simulation parameters can be overridden
		 by the estimated physical parameters $phi$ during optimisation of the objective function. This method updates the physical parameters 
		 (wind, density, stretching and bending stiffness) shared across all time steps (i.e. all parameters except corrective forces).

		Args:
			model_params: physical parameters phi including forces and material properties.

		Returns:
			None
		"""
		surfaces_dir = os.path.join(self.sim_dir, 'surfaces')
		os.makedirs(surfaces_dir, exist_ok=True) 
		arcsim.init_physics(os.path.join(self.sim_dir, 'sim_conf.json'), surfaces_dir, False)
		
		if model_params is not None:
			self.sim.wind.velocity = model_params['wind_velocity']
			self.sim.cloths[0].materials[0].densityori = model_params['density']
			self.sim.cloths[0].materials[0].stretchingori = model_params['stretch']
			self.sim.cloths[0].materials[0].bendingori = model_params['bend']

		
	def run(self, n_frames, correctives=None): 
		"""
		Run physical simulation to generate simulated states given the initial state (template) and the physical parameters. Simulator takes n_steps where n_steps = n_frames * steps_per_frame. Estimated correctives are applied by modifying vertex velocities; velocity[t_th_frame, i_th_vertex] = velocity[t_th_frame, i_th_vertex] + corrective_force[t_th_frame, i_th_vertex]

		Args:
			n_frames: Number of input frames/time steps
			correctives_vert: [n_frames, n_vertices, 3] External velocities 
		Returns:
			meshes_verts: [n_frames, n_vertices, 3] Reconstructed mesh vertices location
		"""
		n_steps = n_frames * self.frame_steps - 1
		meshes_verts = [] 
		for step in range(n_steps):
			if correctives is not None:
				for vert_idx in range(correctives.shape[1]):
					self.sim.cloths[0].mesh.nodes[vert_idx].v += (correctives[self.sim.frame, vert_idx] / self.frame_steps)
			mesh_verts = []
			if step % self.frame_steps == 0: 
				for node in self.sim.cloths[0].mesh.nodes: 
					mesh_verts.append(node.x)	
				mesh_verts = torch.stack(mesh_verts, dim=0).to(self.device)
				meshes_verts.append(mesh_verts)
			arcsim.sim_step()
		meshes_verts = torch.stack(meshes_verts, dim=0)	
		return meshes_verts
