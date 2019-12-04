import numpy as np
from PIL import Image

# Number of rows/columns in elevation data image
ELEVATION_DATA_DIM = 30400
# Scaling factor for elevation data (to give elevation in meters)
ELEVATION_SCALING_FACTOR = 0.5
# resolution of the elevation data, meters/pixel
MAP_RESOLUTION = 20
# Cutoff level for masking PSRs
PSR_RED_CUTOFF = 120
PSR_OFF_CUTOFF = 50

print("Adding psr regions to green channel of composite map...")


def load_psrs_as_array(shape):
    # import image with red mask over PSRs (from Mazarico et al. 2011)
    psr_map = Image.open('mazarico-2011.png')
    psr_map = psr_map.resize(shape[:2])
    psr_map = np.array(psr_map)

    # extract mask of where the PSRs are (large ones marked in red)
    red_channel_psr_map = psr_map[:, :, 0]
    green_channel_psr_map = psr_map[:, :, 1]
    blue_channel_psr_map = psr_map[:, :, 2]
    large_psr_mask = ((red_channel_psr_map > PSR_RED_CUTOFF) &
                      (green_channel_psr_map < PSR_OFF_CUTOFF) &
                      (blue_channel_psr_map < PSR_OFF_CUTOFF))
    small_psr_mask = ((red_channel_psr_map < PSR_OFF_CUTOFF) &
                      (green_channel_psr_map > PSR_RED_CUTOFF) &
                      (blue_channel_psr_map > PSR_RED_CUTOFF))
    psr_mask = large_psr_mask | small_psr_mask

    return psr_mask


def add_psrs_to_composite(composite_img):
    psr_mask = load_psrs_as_array(composite_img.shape)
    green_channel_psr_mask = np.zeros(shape=composite_img.shape,
                                      dtype=np.bool_)
    green_channel_psr_mask[:, :, 1] = psr_mask
    print("PSR mask read from file.")
    composite_img[green_channel_psr_mask] = 1


composite_img = np.load('80S_20M_composite_map_h30_r1000.npy', mmap_mode='r+')
print("Composite read from file.")

add_psrs_to_composite(composite_img)

# save
# np.save("80S_20M_composite_map_h30_r1000", composite_img)
