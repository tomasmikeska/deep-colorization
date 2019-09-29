import numpy as np
from keras.callbacks import Callback
from dataset import read_image
from skimage.color import rgb2gray


class LogImages(Callback):
    '''Log colorized images to CometML'''

    def __init__(self,
                 experiment,
                 paths=[],
                 model=None,
                 img_size=(256, 256),
                 log_iters=1000):
        '''Callback initializer

        Args:
            experiment (CometMl Experiment): CometML Experiment instance
            paths ([string]): List of paths to images that will be colorized and uploaded
            img_size (tuple): Image size tuple in format (width, height)
            log_iters (int): Number of batches after which callback is run
        '''
        self.experiment = experiment
        self.paths      = paths
        self.img_size   = img_size
        self.log_iters  = log_iters
        self.iter       = 0
        if model:
            self.model = model

    def on_batch_end(self, batch, logs):
        self.iter += 1
        if self.iter % self.log_iters == 0:
            self.log_colorized_images(self.iter)

    def log_colorized_images(self, iter):
        ground_truth = []
        batch = []
        # Read images
        for path in self.paths:
            img_rgb = read_image(path, self.img_size)
            ground_truth.append(img_rgb)
            img_gray = rgb2gray(img_rgb).reshape(self.img_size + (1,))
            batch.append(img_gray)
        # Predict color using trained model
        colorized = self.model.predict(np.repeat(batch, 3, axis=-1))
        # Concat ground truth and predicted images and log them to comet
        for i in range(len(colorized)):
            final = np.concatenate((ground_truth[i], colorized[i]), axis=1)
            final = np.rint(final * 127.5 + 127.5).astype(np.uint8)
            self.experiment.log_image(final, name=f'iter_{iter:06d}_image_{i:02d}')
