"""
Written by C. Dawson (cbd@mit.edu)

Reads in topographical data of the lunar south pole region
and computes a map of all locations where a tower could see into
a PSR.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Number of rows/columns in elevation data image
ELEVATION_DATA_DIM = 30400
# Scaling factor for elevation data (to give elevation in meters)
ELEVATION_SCALING_FACTOR = 0.5
# resolution of the elevation data, meters/pixel
MAP_RESOLUTION = 20
# Cutoff level for masking PSRs
PSR_RED_CUTOFF = 200
PSR_OFF_CUTOFF = 50
# height of tower, meters
TOWER_HEIGHT = 30
# max range of tower from PSR, meters
TOWER_OPERATIONAL_RADIUS = 1000
# and in pixels
PIXEL_RADIUS = TOWER_OPERATIONAL_RADIUS / MAP_RESOLUTION


def get_line(x1, y1, x2, y2):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]

    http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm#Python
    """
    # Setup initial conditions
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


# utility function to convert pixel locations in a one image (shape1) to
# pixel locations in a another image (shape2)
def convert_pixel_locations(row, col, shape1, shape2):
    row_scale = shape2[0] / shape1[0]
    col_scale = shape2[1] / shape1[1]

    scaled_location = (row * row_scale, col * col_scale)

    return (round(loc) for loc in scaled_location)


print("Starting landing site evaluation...")

# import image with red mask over PSRs (from Mazarico et al. 2011)
psr_map = Image.open('mazarico-2011.png')
# psr_map = psr_map.resize((ELEVATION_DATA_DIM, ELEVATION_DATA_DIM))
psr_map = np.array(psr_map)
# extract mask of where the PSRs are (large ones marked in red, small in cyan)
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
print("PSR mask read from file.")


# Import the elevation data
south_pole_topo = np.memmap('LDEM_80S_20M.IMG', np.int16).astype(np.float32)
south_pole_topo = south_pole_topo.reshape(ELEVATION_DATA_DIM,
                                          ELEVATION_DATA_DIM)
south_pole_topo *= ELEVATION_SCALING_FACTOR

# crop the elevation data to match the PSR image
crop_half_width = 260
width = south_pole_topo.shape[1]
left = round(width / 2 - crop_half_width * 1000 / MAP_RESOLUTION)
right = round(width / 2 + crop_half_width * 1000 / MAP_RESOLUTION)

height = south_pole_topo.shape[0]
top = round(height / 2 - crop_half_width * 1000 / MAP_RESOLUTION)
bottom = round(height / 2 + crop_half_width * 1000 / MAP_RESOLUTION)

south_pole_topo = south_pole_topo[top:bottom, left:right]

print("Elevation data read from file.")

# # test data
# psr_mask = np.array([
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 1]
# ])
# south_pole_topo = np.array([
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1000, 1000],
#     [0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0]
# ])

# monitor progress by counting how many psr_pixels we have
num_psr_pixels = float(np.sum(psr_mask))
num_psr_pixels_enumerated = 0
print("Starting to enumerate PSRs ({} PSR pixels to enumerate!)".format(
    num_psr_pixels
))

# We want to construct a new map where each point is an integer counting how
# many pixels in a PSR that it can see
visibility_map = np.zeros(south_pole_topo.shape).astype(np.int16)
for psr_y_unscaled, psr_row in enumerate(psr_mask):
    for psr_x_unscaled, pixel in enumerate(psr_row):
        # skip pixels outside of PSR
        if not pixel:
            continue

        # scale up the location to match the scale of the larger map
        psr_y, psr_x = convert_pixel_locations(psr_y_unscaled, psr_x_unscaled,
                                               psr_mask.shape,
                                               south_pole_topo.shape)

        # for pixels inside the PSR, we want to add one to each pixel in
        # the visibility map that is
        #
        #  a) within TOWER_OPERATIONAL_RADIUS km of the PSR pixel
        #  b) has line of sight to the PSR pixel

        # to save cost, we only consider pixels within the radius
        start_row = max(0, psr_y - int(PIXEL_RADIUS))
        end_row = min(ELEVATION_DATA_DIM, psr_y + int(PIXEL_RADIUS))
        start_col = max(0, psr_x - int(PIXEL_RADIUS))
        end_col = min(ELEVATION_DATA_DIM, psr_x + int(PIXEL_RADIUS))

        # iterate through pixels in the bounding box, checking if they're in
        # the circle of the right radius

        # precompute the tower operational radius squared (in pixels)
        PIXEL_RADIUS_SQ = PIXEL_RADIUS**2
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                if (row - psr_y)**2 + (col - psr_x)**2 > PIXEL_RADIUS_SQ:
                    continue

                # also skip all cells in the psr itself
                row_scaled, col_scaled = convert_pixel_locations(
                    row, col, south_pole_topo.shape, psr_mask.shape
                )
                row_scaled = min(row_scaled, psr_mask.shape[0] - 1)
                col_scaled = min(col_scaled, psr_mask.shape[1] - 1)
                if psr_mask[row_scaled, col_scaled]:
                    continue

                # otherwise, check line of sight
                unobstructed = True  # set to false if line of sight obstructed
                line_to_check = get_line(row, col, psr_y, psr_x)
                # slope for line of sight
                distance_to_psr = np.sqrt((row - psr_y)**2 + (col - psr_x)**2)
                slope = (south_pole_topo[row, col] -
                         south_pole_topo[psr_y, psr_x])
                slope /= distance_to_psr
                for coord_to_check in line_to_check:
                    distance_to_check = np.sqrt((row - coord_to_check[0])**2 +
                                                (col - coord_to_check[1])**2)
                    elevation = south_pole_topo[coord_to_check[0],
                                                coord_to_check[1]]
                    line_of_sight_elevation = south_pole_topo[row, col] - \
                        slope * distance_to_check
                    if line_of_sight_elevation < elevation:
                        unobstructed = False
                        break

                # if we're not obstructed, add a tick to the visibility map
                if unobstructed:
                    # print("PSR Pixel {} can be seen by tower at {}".format(
                    #     (psr_x, psr_y), (row, col)
                    # ))
                    visibility_map[row, col] += 1

        num_psr_pixels_enumerated += 1
        print("\r\tEnumerated {:.3f}% of PSR pixels".format(
            num_psr_pixels_enumerated / num_psr_pixels * 100), end='')
print("\nDone enumerating PSR pixels")

# plt.figure(1)
# test_range_start, test_range_end = convert_pixel_locations(500, 600, psr_mask.shape, south_pole_topo.shape)
# plt.imshow(south_pole_topo[test_range_start:test_range_end, test_range_start:test_range_end])
# plt.figure(2)
# plt.imshow(visibility_map[test_range_start:test_range_end, test_range_start:test_range_end])
# plt.figure(3)
# plt.imshow(psr_mask)

# plt.show()

# make sure to save the visibility map
np.save("80S_20M_visibility_map_h{}_r{}".format(
    TOWER_HEIGHT, TOWER_OPERATIONAL_RADIUS
), visibility_map)
