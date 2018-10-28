import numpy as np
import numpy.ma
import scipy.misc
import matplotlib.pyplot as plt

pixel_value = 130   #Value in range 0 to 255

image = scipy.misc.imread("../Downloads/dip.jpg")

#Extract Blue, Green, and Red image from original image
image_B = numpy.copy(image[:, :, 0])
image_G = numpy.copy(image[:, :, 1])
image_R = numpy.copy(image[:, :, 2])

#Define mask depending on pixel value in Red image
image_mask = numpy.empty([image.shape[0], image.shape[1]], dtype = bool)
image_mask[image_R < pixel_value] = False

# #Apply mask to Blue, Green, and Red images
# B_masked = numpy.ma.masked_array(image_B, mask = ~image_mask)
# G_masked = numpy.ma.masked_array(image_G, mask = ~image_mask)
# R_masked = numpy.ma.masked_array(image_R, mask = ~image_mask)
#
# #Stack masked images together again
# masked_image = numpy.ma.dstack((B_masked, G_masked, R_masked))

# create mask with same dimensions as image
mask = numpy.zeros_like(image)

# copy your image_mask to all dimensions (i.e. colors) of your image
for i in range(3):
    mask[:,:,i] = image_mask.copy()

# apply the mask to your image
masked_image = np.multiply(image, mask)

#Plot original image and masked version
fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
ax1.imshow(image)

ax2 = fig.add_subplot(2, 1, 2)
ax2.imshow(masked_image)

plt.show()
