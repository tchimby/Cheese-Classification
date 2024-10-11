import hydra
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
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


class TestDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        self.images_list = [image for image in images_list if image.endswith(".jpg")]

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.test_transform(image)
        return image, os.path.splitext(image_name)[0]

    def __len__(self):
        return len(self.images_list)


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@hydra.main(config_path="configs/train", config_name="config")
def create_submission(cfg):
    test_dataset = TestDataset(cfg.dataset.test_path, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False)

    class_names = sorted(os.listdir(cfg.dataset.train_path))

    graph_def = tf.compat.v1.GraphDef()
    labels = []

    # Load TensorFlow model
    filename = "/Data/hala.gamouh/cheese_classification_challenge/model.pb"
    labels_filename = "/Data/hala.gamouh/cheese_classification_challenge/labels.txt"

    with tf.io.gfile.GFile(filename, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with open(labels_filename, 'rt') as lf:
        for l in lf:
            labels.append(l.strip())

    with tf.compat.v1.Session() as sess:
        input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
        network_input_size = input_tensor_shape[1]

        submission = pd.DataFrame(columns=["id", "label"])

        for images, image_names in test_loader:
            print(f"Processing batch with {len(images)} / {len(image_names)}")

            for image, image_name in zip(images, image_names):
                # Preprocess the image
                image_path = os.path.join(cfg.dataset.test_path, image_name + '.jpg')
                image = Image.open(image_path)

                image = update_orientation(image)

                image = convert_to_opencv(image)

                image = resize_down_to_1600_max_dim(image)
                h, w = image.shape[:2]
                min_dim = min(w,h)
                max_square_image = crop_center(image, min_dim, min_dim)
                augmented_image = resize_to_256_square(max_square_image)
                network_input_size = input_tensor_shape[1]

                augmented_image = crop_center(augmented_image, network_input_size, network_input_size)
                output_layer = 'loss:0'
                input_node = 'Placeholder:0'
                prob_tensor = sess.graph.get_tensor_by_name(output_layer)
                predictions = sess.run(prob_tensor, {input_node: [augmented_image] })
                highest_probability_index = np.argmax(predictions)
                predicted_label = labels[highest_probability_index]

                submission = pd.concat(
                    [
                        submission,
                        pd.DataFrame({"id": [image_name], "label": [predicted_label]}),
                    ]
                )

        submission.to_csv("/Data/hala.gamouh/cheese_classification_challenge/submission_ipadapt_tf2_4hrs_dalle.csv", index=False)


if __name__ == "__main__":
    create_submission()
