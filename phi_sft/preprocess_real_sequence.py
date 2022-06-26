import os
import sys

from geometry.template import generate_template_surface, clean_template_surface
from image.blur import blur_masked_images
from utils.configparser import config_parser
from utils.device import get_device

'''
python preprocess_real_sequence.py  ${data_root}/real ${sequence_name} blur_masked_images 
python preprocess_real_sequence.py  ${data_root}/real ${sequence_name} generate_template_surface
python preprocess_real_sequence.py  ${data_root}/real ${sequence_name} clean_template_surface
'''

if __name__ == "__main__":
    data_dir = sys.argv[1] 
    sequence_name = sys.argv[2] 
    preprocess_action = sys.argv[3] 

    sequence_dir = os.path.join(data_dir, sequence_name) 
    args_data = config_parser(os.path.join(sequence_dir, 'preprocess.ini'))['DEFAULT']

    device = get_device()

    if preprocess_action == 'generate_template_surface':
        generate_template_surface(sequence_dir, device)
    elif preprocess_action == 'clean_template_surface':
        clean_template_surface(sequence_dir, args_data, device)
    elif preprocess_action == 'blur_masked_images':
        blur_masked_images(sequence_dir, args_data, device)