"""
Written by C. Dawson (cbd@mit.edu)

Reads in topographical data of the lunar south pole region
and displays a map.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


print("Creating topo/visibility composite map...")

# # import image with red mask over PSRs (from Mazarico et al. 2011)
# psr_map = Image.open('mazarico-2011.png')
# # psr_map = psr_map.resize((ELEVATION_DATA_DIM, ELEVATION_DATA_DIM))
# psr_map = np.array(psr_map)
# # extract mask of where the PSRs are (large ones marked in red)
# red_channel_psr_map = psr_map[:, :, 0]
# green_channel_psr_map = psr_map[:, :, 1]
# blue_channel_psr_map = psr_map[:, :, 2]
# large_psr_mask = ((red_channel_psr_map > PSR_RED_CUTOFF) &
#                   (green_channel_psr_map < PSR_OFF_CUTOFF) &
#                   (blue_channel_psr_map < PSR_OFF_CUTOFF))
# small_psr_mask = ((red_channel_psr_map < PSR_OFF_CUTOFF) &
#                   (green_channel_psr_map > PSR_RED_CUTOFF) &
#                   (blue_channel_psr_map > PSR_RED_CUTOFF))
# psr_mask = large_psr_mask | small_psr_mask
# print("PSR mask read from file.")

# Import the elevation data
# use functions to save memory ;)
def create_composite_with_topo():
    south_pole_topo = np.memmap('LDEM_80S_20M.IMG', np.int16).astype(np.float32)
    south_pole_topo = south_pole_topo.reshape(ELEVATION_DATA_DIM,
                                              ELEVATION_DATA_DIM)
    south_pole_topo *= ELEVATION_SCALING_FACTOR

    # crop
    crop_half_width = 260
    width = south_pole_topo.shape[1]
    left = round(width / 2 - crop_half_width * 1000 / MAP_RESOLUTION)
    right = round(width / 2 + crop_half_width * 1000 / MAP_RESOLUTION)

    height = south_pole_topo.shape[0]
    top = round(height / 2 - crop_half_width * 1000 / MAP_RESOLUTION)
    bottom = round(height / 2 + crop_half_width * 1000 / MAP_RESOLUTION)

    south_pole_topo = south_pole_topo[top:bottom, left:right]
    print("Elevation data read from file.")

    # We want the red channel to be topo, the green to be psr,
    # and the blue to be visibility. For now, we only include the
    # topo and visibility (we'll add psrs in the next step when we have
    # more memory)
    composite_img = np.zeros(shape=tuple(list(south_pole_topo.shape) + [3]),
                             dtype=np.float32)
    composite_img[:, :, 0] = south_pole_topo

    return composite_img


# import visibility map
visibility_map = np.load('80S_20M_visibility_map_h30_r1000.npy')
print("Visibility data read from file.")

composite_img = create_composite_with_topo()
composite_img[:, :, 2] = visibility_map.astype(np.float32)

# save
np.save("80S_20M_composite_map_h30_r1000", composite_img)

print("Composite saved!")
