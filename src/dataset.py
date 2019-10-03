import numpy as np
import skimage
import random
from itertools import cycle
from functools import partial
from PIL import Image
from sklearn.utils import shuffle
from skimage.color import rgb2gray
from keras.utils import to_categorical, Sequence
from keras.preprocessing.image import ImageDataGenerator
from utils import dir_listing, file_listing, take


def resize_keep_ratio(img_np, size):
    '''
    Resize image to specified size while keeping the aspect ratio. Image is pasted on black background.

    Args:
        img_np (numpy array): Image numpy array in format (h, w, c).
        size (tuple): Tuple in format (w, h)
    '''
    try:
        source_image = Image.fromarray(img_np)
        orig_w, orig_h = source_image.size
        if orig_w < size[0] and orig_h < size[1]:
            return None
        final_image = Image.new('RGB', size, 'black')
        source_image.thumbnail(size)
        w, h = source_image.size
        final_image.paste(source_image, (int((size[0] - w) / 2), int((size[1] - h) / 2)))
        return np.array(final_image)
    except Exception:
        return None


def read_image(path, size=None, keep_ratio=False):
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
        img_np = skimage.io.imread(path)
        if size and keep_ratio:
            img_np = resize_keep_ratio(img_np)
        elif size and not keep_ratio:
            img_np = skimage.transform.resize(img_np, size) * 255
        if img_np.ndim == 2 or img_np.shape[2] == 1:
            img_np = np.reshape(img_np, (img_np.shape[0], img_np.shape[1], 1))
            img_np = np.repeat(img_np, 3, axis=-1)
        return img_np / 127.5 - 1
    except Exception:
        return None


def add_noise(X, noise_count=None, noise_range=30):
    '''
    Add randomly distributed noise to images

    Args:
        X (numpy): batch of images
        noise_count (number): Number of pixels in each image to change
        noise_range (number): Range of random values to add/subtract from selected pixels
    '''
    for img_np in X:
        for _ in range(noise_count if noise_count else img_np.shape[0] * 2):
            xx = random.randrange(img_np.shape[1])
            yy = random.randrange(img_np.shape[0])

            img_np[yy][xx][0] += random.randrange(-noise_range, noise_range) / 127.5
            img_np[yy][xx][1] += random.randrange(-noise_range, noise_range) / 127.5
            img_np[yy][xx][2] += random.randrange(-noise_range, noise_range) / 127.5

    return X


def take(n, arr, extract_fn=None):
    '''
    Helper to take N elements from array, skipping None

    Args:
        n (int): Number of elements to take
        arr (list): List of elements
        extract_fn (function): Function that is used to transform item from array, takes item as the only parameter
    '''
    output = []
    count = 0

    for item in cycle(arr):
        if item is None:
            continue
        if extract_fn:
            item = extract_fn(item)
        output.append(item)
        count += 1
        if count >= n:
            return output


def read_batch(idx, paths, batch_size=32, img_size=(48, 48)):
    '''
    Read batch tuple (X, y) on index idx from paths. X are grayscale images, y are target RGB images

    Args:
        idx (number): Index of batch to take
        paths (list): List of paths to read
        batch_size (number): Batch size - number of tuples to read
        img_size (tuple): Images size after squarification, in (W, H) format
    '''
    X, y = [], []

    for img_rgb in take(batch_size, paths[idx * batch_size:], extract_fn=partial(read_image, size=img_size)):
        img_gray = rgb2gray(img_rgb).reshape(img_size + (1,))
        X.append(np.repeat(img_gray, 3, axis=-1))
        y.append(img_rgb)

    return np.array(X), np.array(y)


class TrainDatasetSequence(Sequence):
    def __init__(self,
                 base_train_path,
                 batch_size=128,
                 img_size=(256, 256)):
        self.batch_size = batch_size
        self.paths      = shuffle(self._get_image_paths(base_train_path))
        self.img_size   = img_size

    def _get_image_paths(self, base_path):
        image_paths = []
        for dirpath in dir_listing(base_path):
            image_paths += file_listing(dirpath, extension='jpg')
        return image_paths

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, idx):
        X, y = read_batch(idx, self.paths, self.batch_size, self.img_size)
        X = add_noise(X)  # Add noise to stabilize results
        return X, y


class TestDatasetSequence(Sequence):
    def __init__(self,
                 base_test_path,
                 batch_size=128,
                 img_size=(256, 256)):
        self.batch_size = batch_size
        self.paths      = file_listing(base_test_path, extension='JPEG')
        self.img_size   = img_size

    def __len__(self):
        return len(self.paths) // self.batch_size

    def __getitem__(self, idx):
        return read_batch(idx, self.paths, self.batch_size, self.img_size)
