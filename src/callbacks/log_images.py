import numpy as np
from keras.callbacks import Callback
from dataset import read_image, lab_to_rgb, rgb_to_lab


class LogImages(Callback):
    '''Log colorized images to CometML'''

    def __init__(self, experiment, paths=[], img_size=(256, 256)):
        self.experiment = experiment
        self.paths = paths
        self.img_size = img_size

    def on_epoch_end(self, epoch, logs):
        ground_truth = []
        batch = []
        # Read images
        for path in self.paths:
            img_rgb = read_image(path, self.img_size)
            img_lab = rgb_to_lab(img_rgb)
            ground_truth.append(img_rgb)
            batch.append(img_lab[:,:,:1])
        # Predict AB channels
        preds = self.model.predict(np.array(batch))
        # Concat ground truth and predicted images and log them to comet
        for i in range(len(preds)):
            colorized = lab_to_rgb(batch[i], preds[i])
            final = np.concatenate((ground_truth[i], colorized), axis=1)
            final = np.rint(final * 127.5 + 127.5).astype(np.uint8)
            self.experiment.log_image(final, name=f'epoch_{epoch:02d}_image_{i:02d}')
