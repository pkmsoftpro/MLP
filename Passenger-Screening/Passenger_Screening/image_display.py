import numpy as np
import matplotlib
import tsa_notebook as tsa

image = tsa.read_data('stage1_aps_01941f33fd090ae5df8c95992c027862.aps')
print('original shape: {}'.format(image.shape))

image = image.transpose()
print('after transpose shape: {}'.format(image.shape))

# first image
frame1 = image[0]
print('frame1 shape: {}'.format(frame1.shape))

# plot the first image
fig = matplotlib.pyplot.figure(figsize = (8, 8))
fig1 = matplotlib.pyplot.figure(figsize = (8, 8))

ax = fig.add_subplot(2, 2, 1)
# im1 = ax.imshow(np.flipud(image[:, :, 0].transpose()), cmap = 'viridis')
im1 = ax.imshow(frame1, cmap = 'viridis')

ax = fig.add_subplot(2, 2, 2)
frame1 = np.flipud(frame1)
im2 = ax.imshow(frame1, cmap = 'viridis')


ax = fig.add_subplot(2, 2, 3)
frame1_gray = tsa.convert_to_grayscale(frame1)
im3 = ax.imshow(frame1_gray, cmap = 'viridis')

ax = fig.add_subplot(2, 2, 4)
frame1_contrast = tsa.spread_spectrum(frame1_gray)
im4 = ax.imshow(frame1_contrast, cmap = 'viridis')


for i in range(16):
	# sub_loc = int('44{}'.format(i+1))
	ax1 = fig1.add_subplot(4,4,i+1)
	im = ax1.imshow(np.flipud(image[i]), cmap = 'viridis')


matplotlib.pyplot.show()


