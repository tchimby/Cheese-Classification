import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


def convert_to_opencv(image):
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image

def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    startx = w//2 - (cropx//2)
    starty = h//2 - (cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

def resize_down_to_1600_max_dim(image):
    h, w = image.shape[:2]
    if h < 1600 and w < 1600:
        return image

    new_size = (1600 * w // h, 1600) if h > w else (1600, 1600 * h // w)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

def resize_to_256_square(image):
    h, w = image.shape[:2]
    return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

def update_orientation(image):
    exif_orientation_tag = 0x0112
    if hasattr(image, '_getexif'):
        exif = image._getexif()
        if exif is not None and exif_orientation_tag in exif:
            orientation = exif.get(exif_orientation_tag, 1)
            orientation -= 1
            if orientation >= 4:
                image = image.transpose(Image.TRANSPOSE)
            if orientation == 2 or orientation == 3 or orientation == 6 or orientation == 7:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            if orientation == 1 or orientation == 2 or orientation == 5 or orientation == 6:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


# Load the TensorFlow model (trained with Azure Custom Vision Service)
graph_def = tf.compat.v1.GraphDef()
labels = []

filename = "/Data/hala.gamouh/cheese_classification_challenge/model.pb"
labels_filename = "/Data/hala.gamouh/cheese_classification_challenge/labels.txt"

with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())




from PIL import Image
import numpy as np
import cv2

imageFile = "/Data/hala.gamouh/cheese_classification_challenge/dataset/test/0AZMAnAz4qFeevZ.jpg"
image = Image.open(imageFile)

image = update_orientation(image)

# Convert to OpenCV format
image = convert_to_opencv(image)



image = resize_down_to_1600_max_dim(image)
h, w = image.shape[:2]
min_dim = min(w,h)
max_square_image = crop_center(image, min_dim, min_dim)
augmented_image = resize_to_256_square(max_square_image)
with tf.compat.v1.Session() as sess:
    input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
network_input_size = input_tensor_shape[1]

augmented_image = crop_center(augmented_image, network_input_size, network_input_size)
output_layer = 'loss:0'
input_node = 'Placeholder:0'

with tf.compat.v1.Session() as sess:
    try:
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        predictions = sess.run(prob_tensor, {input_node: [augmented_image] })
    except KeyError:
        print ("Couldn't find classification output layer: " + output_layer + ".")
        print ("Verify this a model exported from an Object Detection project.")
        exit(-1)
        
    highest_probability_index = np.argmax(predictions)
    print('Classified as: ' + labels[highest_probability_index])
    print()

