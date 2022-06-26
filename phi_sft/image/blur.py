import os
import pdb
import torchgeometry as tgm

from image.image_io import read_images, save_images

def blur_masked_images(sequence_dir, args_data, device):
    print("blurring masks...")
     
    input_mask_dir = os.path.join(sequence_dir, "masks")
    input_masks = read_images(input_mask_dir, device=device)
    input_masks = input_masks[:,None,...]
    gauss = get_gauss_instance(args_data)
    blurred_masks = gauss(input_masks).squeeze()
    blurred_mask_dir = os.path.join(sequence_dir, 'blurred_masks')
    save_images(blurred_masks, blurred_mask_dir, image_prefix='blurred_mask_')

def get_gauss_instance(args_data):
    kernel_size = args_data.getint('kernel_size')
    blur_radius = args_data.getint('blur_radius')
    gauss = tgm.image.GaussianBlur((kernel_size, kernel_size), (blur_radius, blur_radius))
    return gauss