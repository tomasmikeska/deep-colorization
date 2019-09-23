import os
import argparse
import numpy as np
from comet_ml import Experiment
from tqdm import tqdm
from sklearn.utils import shuffle
from train_gan import build_gan, create_parallel_queue
from dataset import TrainDatasetSequence
from utils import relative_path, file_listing


def train_discriminator(gan,
                        generator,
                        discriminator,
                        gray_imgs,
                        rgb_imgs,
                        batch_size=32):
    colorized_images = generator.predict(gray_imgs)

    batch_X = np.concatenate((rgb_imgs, colorized_images))
    batch_y = np.zeros((batch_size, 1))
    batch_y[:batch_size // 2, :] = 0.9
    batch_X, batch_y = shuffle(batch_X, batch_y)

    return discriminator.train_on_batch(batch_X, batch_y)


def train(args, experiment):
    # Dataset
    train_seq = TrainDatasetSequence(
        args.train_dataset,
        batch_size=args.batch_size // 2,
        img_size=(args.img_w, args.img_h))

    # Build GAN
    gan, generator, discriminator = build_gan(
        (args.img_w, args.img_h),
        args.generator_weights)
    generator.trainable = False
    discriminator.trainable = True

    # Create parallel queue to load images asynchronously
    batch_queue = create_parallel_queue(train_seq)

    # Pre-train discriminator
    for batch_idx in tqdm(range(args.steps)):
        gray_imgs, rgb_imgs = next(batch_queue)
        loss = train_discriminator(
            gan,
            generator,
            discriminator,
            gray_imgs,
            rgb_imgs,
            args.batch_size)
        # Log metrics
        metrics = {
            'loss': loss
        }
        experiment.log_metrics(metrics, step=batch_idx)

    # Save discriminator
    discriminator.save(args.model_save_path + '/discriminator.h5')


if __name__ == '__main__':
    # Command line arguments parsing
    parser = argparse.ArgumentParser(description='Train a colorization deep learning model')
    parser.add_argument('--train-dataset',
                        type=str,
                        default=relative_path('../data/train/'),
                        help='Train dataset base path. Folder should contain subfolder for each class.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        help='Batch size used during training')
    parser.add_argument('--img-w',
                        type=int,
                        default=256,
                        help='Image width')
    parser.add_argument('--img-h',
                        type=int,
                        default=256,
                        help='Image height')
    parser.add_argument('--generator-weights',
                        type=str,
                        help='Pre-trained generator weights')
    parser.add_argument('--discriminator-weights',
                        type=str,
                        help='Pre-trained discriminator weights')
    parser.add_argument('--gan-weights',
                        type=str,
                        help='GAN (generator+discriminator) weights')
    parser.add_argument('--steps',
                        type=int,
                        default=3000,
                        help='Number of steps (batches) to perform')
    parser.add_argument('--model-save-path',
                        type=str,
                        default=relative_path('../model/'),
                        help='Base directory to save model during training')
    args = parser.parse_args()
    # CometML experiment
    experiment = None
    if os.getenv('COMET_API_KEY'):
        experiment = Experiment(api_key=os.getenv('COMET_API_KEY'),
                                project_name=os.getenv('COMET_PROJECTNAME'),
                                workspace=os.getenv('COMET_WORKSPACE'))
    # Train
    train(args, experiment)
