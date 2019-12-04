import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


composite_img = np.load('80S_20M_composite_map_h30_r1000.npy', mmap_mode='r')

# crop_half_width = 50
# width = composite_img[:, :, 0].shape[1]
# left = round(width / 2 - crop_half_width * 1000 / 20)
# right = round(width / 2 + crop_half_width * 1000 / 20)

# height = composite_img[:, :, 0].shape[0]
# top = round(height / 2 - crop_half_width * 1000 / 20)
# bottom = round(height / 2 + crop_half_width * 1000 / 20)

medium_image = composite_img[15000:20000, 15000:20000, :]

fig = plt.figure(0)
terrain = plt.imshow(medium_image[:, :, 0], cmap=cm.terrain)
cbar = fig.colorbar(terrain)
cbar.set_label("Relative elevation above reference sphere")

# plt.xticks(np.linspace(0, medium_image.shape[1], 10), np.round(np.linspace(-260, 260, 10)).astype(np.int))
# plt.yticks(np.linspace(0, medium_image.shape[0], 10), np.round(np.linspace(-260, 260, 10)).astype(np.int))
# plt.xlabel("X (km)")
# plt.ylabel("Y (km)")

psr_mask = medium_image[:, :, 1] < 1
plt.imshow(np.ma.masked_where(psr_mask, medium_image[:, :, 1]), cmap=cm.gray)

vis_mask = medium_image[:, :, 2] > 0.00001
vis_map = medium_image[:, :, 2]
vis_map[vis_mask] = 2000
vis_mask = medium_image[:, :, 2] < 0.00001
plt.imshow(np.ma.masked_where(vis_mask, np.zeros(vis_map.shape)), cmap=cm.cool)

plt.show()
