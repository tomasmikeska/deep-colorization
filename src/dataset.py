import numpy as np
import skimage
import imagesize
from itertools import cycle
from PIL import Image
from sklearn.utils import shuffle
from skimage.color import rgb2gray
from keras.utils import to_categorical, Sequence
from keras.preprocessing.image import ImageDataGenerator
from utils import dir_listing, file_listing, take


def read_image(path, size=(256, 256)):
    '''
    Read image from specified path and resize it if size specified

    Args:
        path (string): Path to image
        size (tuple): Size of output image, tuple of integers in format (W, H)

    Returns:
        Numpy array representing image with values in range (-1, 1)
        or None if image is too small (both sides are smaller than target size)
    '''
    try:
        source_image = Image.open(path).convert('RGB')
        orig_w, orig_h = source_image.size
        if orig_w < size[0] and orig_h < size[1]:
            return None
        final_image = Image.new('RGB', size, 'black')
        source_image.thumbnail(size)
        w, h = source_image.size
        final_image.paste(source_image, (int((size[0] - w) / 2), int((size[1] - h) / 2)))
        return np.array(final_image) / 127.5 - 1
    except Exception:
        return None


def rgb_to_lab(img_np):
    '''
    Convert RGB image to AB channels of CIE L*a*b color space.

    Args:
        img_np (numpy array): RGB image numpy array, shape (w, h, 3), range (-1, 1)

    Returns:
        Numpy array with shape (w, h, 3), format (L, A, B), range (-1, 1)
    '''
    output = np.empty_like(img_np)
    img_lab = skimage.color.rgb2lab(img_np / 2. + 0.5)
    output[:,:,0] = img_lab[:,:,0] / 50. - 1.
    output[:,:,1:] = img_lab[:,:,1:] / 127.
    return output


def lab_to_rgb(img_l, img_ab):
    '''
    Convert a pair of numpy arrays (l channel and ab channels) into an RGB image

    Args:
        img_l (numpy array): Lightness image numpy array, shape (w, h), range (-1, 1)
        img_ab (numpy array): CIE ab channels, shape (w, h), range (-1, 1)
    '''
    lab = np.empty(img_l.shape[:2] + (3,))
    lab[:, :, 0] = np.squeeze(((img_l + 1) * 50))
    lab[:, :, 1:] = img_ab * 127.
    return skimage.color.lab2rgb(lab) * 2 - 1


class TrainDatasetSequence(Sequence):

    def __init__(self, base_train_path, batch_size=128, img_size=None):
        self.batch_size    = batch_size
        self.paths         = shuffle(self._get_image_paths(base_train_path))
        self.img_size      = img_size
        self.augment_image = gen = ImageDataGenerator(rotation_range=5,
                                                      brightness_range=[0.9, 1.1],
                                                      shear_range=0.05,
                                                      zoom_range=0.05,
                                                      horizontal_flip=True).random_transform

    def _get_image_paths(self, base_path):
        image_paths = []
        for dirpath in dir_listing(base_path):
            image_paths += file_listing(dirpath, extension='jpg')
        return image_paths

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, idx):
        X, y = [], []
        count = 0
        for image_path in cycle(self.paths[idx * self.batch_size:]):
            img_rgb = read_image(image_path, size=self.img_size)
            if img_rgb is None:
                continue
            count += 1
            # img_rgb = self.augment_image(img_rgb)
            # img_lab = rgb_to_lab(img_rgb)
            # X.append(np.repeat(img_lab[:,:,:1], 3, axis=-1))
            # y.append(img_lab[:,:,1:])
            img_gray = rgb2gray(img_rgb).reshape(self.img_size + (1,))
            X.append(np.repeat(img_gray, 3, axis=-1))
            y.append(img_rgb)
            if count >= self.batch_size:
                break
        return np.array(X), np.array(y)


class TestDatasetSequence(Sequence):

    def __init__(self, base_test_path, batch_size=128, img_size=None):
        self.batch_size = batch_size
        self.paths      = file_listing(base_test_path, extension='JPEG')
        self.img_size   = img_size

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, idx):
        X, y = [], []
        count = 0
        for image_path in cycle(self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]):
            img_rgb = read_image(image_path, size=self.img_size)
            if img_rgb is None:
                continue
            count += 1
            # img_lab = rgb_to_lab(img_rgb)
            # X.append(np.repeat(img_lab[:,:,:1], 3, axis=-1))
            # y.append(img_lab[:,:,1:])
            img_gray = rgb2gray(img_rgb).reshape(self.img_size + (1,))
            X.append(np.repeat(img_gray, 3, axis=-1))
            y.append(img_rgb)
            if count >= self.batch_size:
                break
        return np.array(X), np.array(y)
