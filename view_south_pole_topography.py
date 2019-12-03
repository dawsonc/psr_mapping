"""
Written by C. Dawson (cbd@mit.edu)

Reads in topographical data of the lunar south pole region
and displays a map.
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
# height of tower, meters
TOWER_HEIGHT = 30
# max range of tower from PSR, meters
TOWER_OPERATIONAL_RADIUS = 1000


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


# import image with red mask over PSRs (from Mazarico et al. 2011)
psr_map = Image.open('mazarico-2011.png')
psr_map = psr_map.resize((ELEVATION_DATA_DIM, ELEVATION_DATA_DIM))
psr_map = np.array(psr_map)
# extract mask of where the PSRs are (marked in red)
red_channel_psr_map = psr_map[:, :, 0]
psr_mask = red_channel_psr_map > PSR_RED_CUTOFF
print("PSR mask read from file.")


# Import the elevation data
south_pole_topo = np.fromfile('LDEM_80S_20M.IMG', np.int16).astype(np.float32)
south_pole_topo = south_pole_topo.reshape(ELEVATION_DATA_DIM,
                                          ELEVATION_DATA_DIM)
south_pole_topo *= ELEVATION_SCALING_FACTOR
print("Elevation data read from file.")

# monitor progress by counting how many psr_pixels we have
num_psr_pixels = float(np.sum(psr_mask))
num_psr_pixels_enumerated = 0
print("Starting to enumerate PSRs")

# We want to construct a new map where each point is an integer counting how
# many pixels in a PSR that it can see
visibility_map = np.zeros(south_pole_topo.shape)
for psr_y, psr_row in enumerate(psr_mask):
    for psr_x, pixel in enumerate(psr_row):
        # skip pixels outside of PSR
        if not pixel:
            continue

        # for pixels inside the PSR, we want to add one to each pixel in
        # the visibility map that is
        #
        #  a) within TOWER_OPERATIONAL_RADIUS km of the PSR pixel
        #  b) has line of sight to the PSR pixel

        # to save cost, we only consider pixels within the radius
        start_row = max(0, psr_y - int(TOWER_OPERATIONAL_RADIUS / MAP_RESOLUTION))
        end_row = min(ELEVATION_DATA_DIM, psr_y + int(TOWER_OPERATIONAL_RADIUS / MAP_RESOLUTION))
        start_col = max(0, psr_x - int(TOWER_OPERATIONAL_RADIUS / MAP_RESOLUTION))
        end_col = min(ELEVATION_DATA_DIM, psr_x + int(TOWER_OPERATIONAL_RADIUS / MAP_RESOLUTION))

        # iterate through pixels in the bounding box, checking if they're in
        # the circle of the right radius

        # precompute the tower operational radius squared (in pixels)
        TOWER_RAD_SQ_PIXELS = TOWER_OPERATIONAL_RADIUS**2 / MAP_RESOLUTION**2
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                if (row - psr_y)**2 + (col - psr_x)**2 > TOWER_RAD_SQ_PIXELS:
                    continue

                # also skip all cells in the psr itself
                if psr_mask[row, col]:
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
                    line_of_sight_elevation = south_pole_topo[row, col] - slope * distance_to_check
                    if line_of_sight_elevation < elevation:
                        unobstructed = False
                        break

                # if we're not obstructed, add a tick to the visibility map
                if not unobstructed:
                    visibility_map[row, col] += 1

        num_psr_pixels_enumerated += 1
        print("\r\tEnumerated {:.3f}% of PSR pixels".format(
            num_psr_pixels_enumerated / num_psr_pixels * 100
            ), end='')
print("\nDone enumerating PSR pixels")

plt.figure(1)
plt.imshow(south_pole_topo)
plt.figure(2)
plt.imshow(visibility_map)
plt.figure(3)
plt.imshow(psr_mask)

plt.show()
