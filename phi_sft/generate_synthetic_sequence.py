import sys
import pdb
import os
import torch

from utils.configparser import config_parser
from utils.device import get_device
from simulation.physics_simulator import PhysicsSimulator
from render.diff_renderer import DifferentiableRenderer

'''
python generate_synthetic_sequence.py ${data_root}/synthetic ${sequence_name} generate_surfaces
python generate_synthetic_sequence.py ${data_root}/synthetic ${sequence_name} render_surfaces
'''

def generate_surfaces(args_data):
    physics_sim = PhysicsSimulator(args_data['sequence_dir'], device)
    physics_sim.reset()
    n_frames = args_data.getint('n_frames')
    physics_sim.run(n_frames)

def render_surfaces(args_expt, args_data):
    diff_renderer = DifferentiableRenderer(args_expt, args_data, device)
    diff_renderer.render_rgba_vis(args_data['sequence_dir'])

if __name__ == "__main__":

    data_dir = sys.argv[1] 
    sequence_name = sys.argv[2] 
    preprocess_action = sys.argv[3]
    device = get_device()
 
    args_all = config_parser(os.path.join(data_dir, sequence_name, 'preprocess.ini'))
    args_data = args_all['DEFAULT']
    
    if preprocess_action == 'generate_surfaces':
        generate_surfaces(args_data)
    elif preprocess_action =='render_surfaces':
        args_expt = args_all['RENDER']
        render_surfaces(args_expt, args_data)
