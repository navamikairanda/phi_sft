import os
import sys
import pdb

from utils.configparser import config_parser
from utils.device import get_device
from evaluation.metrics import Metrics
from evaluation.align import Align
from render.diff_renderer import DifferentiableRenderer

'''
sequence_type=real
sequence_name=s3
converged_iteration=300

python evaluate_reconstructed_surfaces.py ${data_root}/${sequence_type} config/expt_${sequence_type}_${sequence_name}.ini $converged_iteration
'''

if __name__ == "__main__":
    data_dir = sys.argv[1] 
    args_expt_file = sys.argv[2]
    converged_iter = sys.argv[3]

    args_expt = config_parser(args_expt_file)['DEFAULT'] 
    sequence_dir = os.path.join(data_dir, args_expt['sequence_name']) 
    args_data = config_parser(os.path.join(sequence_dir, 'preprocess.ini'))['DEFAULT']

    device = get_device()
    log_file = os.path.join(args_expt['log_dir'], 'log.txt')
    log = open(log_file, 'a+', buffering=1)
    log.write('\n##### START EVALUATION #######\n')

    recon_dir = os.path.join(args_expt['log_dir'], converged_iter)
    
    align = Align(args_expt, args_data, device) 
    aligned_meshes = align.align_recon_2_gt(os.path.join(recon_dir, 'surfaces'), white_verts_features=True)
    
    # Quantitative evaluation
    metrics = Metrics(args_expt, args_data, log, device) 
    metrics.compute_recon_errors(aligned_meshes)
    evaluation_dir = os.path.join(recon_dir, 'evaluation') if args_expt.getboolean('per_frame_registration') else os.path.join(recon_dir, 'evaluation_noicp')
    
    # Qualitative evaluation
    diff_renderer = DifferentiableRenderer(args_expt, args_data, device, evaluation=True)
    # Input View
    diff_renderer.render_mesh_vis(evaluation_dir, aligned_meshes, False)   
    # Novel View
    diff_renderer.render_mesh_vis(evaluation_dir, aligned_meshes, True)
    
    log.write('\n##### END EVALUATION #######\n')