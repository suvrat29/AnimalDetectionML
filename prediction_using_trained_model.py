import keras
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

classifier = keras.models.load_model('E:/Machine Learning PSL/Python ML modules/trainedmodel/catsanddogsclassifierV2.H5')

test_image = image.load_img('E:/Machine Learning PSL/Python ML modules/cat_dog_dataset/single_prediction/predict4.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
