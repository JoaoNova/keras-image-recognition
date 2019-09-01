import matplotlib.pyplot as plt  # Just used to show images
import numpy as np
from skimage.transform import resize
from keras.models import load_model  # Used to load our already trained model
model = load_model('models/model1.h5')

my_image = plt.imread('img/dog3.jpg')
my_image_resized = resize(my_image, (32, 32, 3))  # Resizing image to 32x32 pixels with RGB
# img = plt.imshow(my_image_resized) # Just used to show resized image
# plt.show()

probabilities = model.predict(np.array([my_image_resized]))
number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Organize probabilities in crescent order
index = np.argsort(probabilities[0, :])

print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0, index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0, index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0, index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0, index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0, index[5]])
