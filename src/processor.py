import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import color
from scipy import interpolate

# read image from src directory
im = io.imread('../data/campus.tiff')

# scaling
black = 150
white = 4095
# multipliers
r_scale = 2.394531
g_scale = 1.000000
b_scale = 1.597656

fig = plt.figure()
plt.imshow(im)


### PYTHON INITIALS ###
(height, width) = np.shape(im)
print("image dimensions: " + str(width) + " x " + str(height) + " with " + str(im.dtype) + " bits per pixel")
im_f = im.astype(np.double)


### LINEARIZATION ###
im_f = (im_f - black) / (white - black)
im_f = np.clip(im_f, 0, 1)
fig = plt.figure()
plt.imshow(im_f)


### DEMOSAICING AND ###
### IDENTIFYING THE CORRECT BAYER PATTERN ###
odd_x = np.arange(1, width, 2)
odd_y = np.arange(1, height, 2)
even_x = np.arange(0, width, 2)
even_y = np.arange(0, height, 2)

red_z = im_f[0::2, 0::2]
blue_z = im_f[1::2, 1::2]
green_z1 = im_f[0::2, 1::2]
green_z2 = im_f[1::2, 0::2]

red_f = interpolate.interp2d(even_x, even_y, red_z)
blue_f = interpolate.interp2d(odd_x, odd_y, blue_z)
green_f1 = interpolate.interp2d(even_x, odd_y, green_z1)
green_f2 = interpolate.interp2d(odd_x, even_y, green_z2)

all_x = np.arange(0, width, 1)
all_y = np.arange(0, height, 1)
im_red = red_f(all_x, all_y)
im_blue = blue_f(all_x, all_y)
im_green1 = green_f1(all_x, all_y)
im_green2 = green_f2(all_x, all_y)
im_green = (im_green1 + im_green2) * 0.5

im_rgb = np.dstack((im_red, im_green, im_blue))

fig = plt.figure()
plt.imshow(im_rgb)


### WHITE BALANCING ###
# white world
r_max = im_red.max()
b_max = im_blue.max()
g_max = im_green.max()

im_red_white = im_red * (g_max / r_max)
im_green_white = im_green
im_blue_white = im_blue * (g_max / b_max)

im_rgb_white = np.dstack((im_red_white, im_green_white, im_blue_white))
fig = plt.figure()
plt.imshow(im_rgb_white)


# grey world
r_avg = im_red.mean()
b_avg = im_blue.mean()
g_avg = im_green.mean()

im_red_grey = im_red * (g_avg / r_avg)
im_green_grey = im_green
im_blue_grey = im_blue * (g_avg / b_avg)

im_rgb_grey = np.dstack((im_red_grey, im_green_grey, im_blue_grey))
fig = plt.figure()
plt.imshow(im_rgb_grey)


# multiply
im_red_mult = im_red * r_scale
im_green_mult = im_green * g_scale
im_blue_mult = im_blue * b_scale

im_rgb_mult = np.dstack((im_red_mult, im_green_mult, im_blue_mult))
fig = plt.figure()
plt.imshow(im_rgb_mult)


### COLOR SPACE CORRECTION ###
M_srgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])

M_xyz_to_cam = np.array([[0.6988, -0.1384, -0.0714],
                         [-0.5631, 1.3410, 0.2447],
                         [-0.1485, 0.2204, 0.7318]])

M_srgb_to_cam = np.matmul(M_xyz_to_cam, M_srgb_to_xyz)
# normalize and invert
row_norms = np.sum(M_srgb_to_cam, axis=1)
M_srgb_to_cam = np.divide(M_srgb_to_cam.T, row_norms).T
invM_srgb_to_cam = np.linalg.inv(M_srgb_to_cam)

im_srgb = np.zeros((height, width, 3))
for i in range(3):
    m_i = invM_srgb_to_cam[i]
    im_srgb[:,:,i] = np.dot(im_rgb_mult, m_i)

fig = plt.figure()
plt.imshow(im_srgb)


### BRIGHTNESS ADJUSTMENT AND GAMMA ENCODING ###

grayscale_intensity = color.rgb2gray(im_srgb).mean()
im_bright = im_srgb * (0.15 / grayscale_intensity)
fig = plt.figure() # create a new figure
plt.imshow(im_bright)

C = im_bright
C = np.where(C <= 0.0031308, 12.92 * C, 1.055 * (C ** (1.0 / 2.4)) - 0.055)
im_gamma = C
fig = plt.figure()
plt.imshow(im_gamma)


### COMPRESSION ###
im_final_float = np.clip(im_gamma, 0, 1) * 255
im_final = im_final_float.astype(np.ubyte)
io.imsave('output.png', im_final)
io.imsave('output.jpeg', im_final, quality=95)