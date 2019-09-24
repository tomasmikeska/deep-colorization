import os
import argparse
import numpy as np
from comet_ml import Experiment
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.metrics import categorical_accuracy
from keras.utils.data_utils import OrderedEnqueuer
from losses.perceptual_loss import perceptual_loss
from dataset import TrainDatasetSequence, TestDatasetSequence
from models.att_unet import AttentionUNet
from models.att_discriminator import AttentionDiscriminator
from callbacks.log_images import LogImages
from utils import relative_path, file_listing


def build_gan(img_size,
              generator_weights=None,
              discriminator_weights=None,
              gan_weights=None):
    input_shape = img_size + (3,)
    img_input  = Input(shape=input_shape)
    # Create generator
    generator = AttentionUNet(input_shape)
    if generator_weights:
        generator.load_weights(generator_weights)
    # Create discriminator
    discriminator = AttentionDiscriminator(input_shape)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=Adam(0.0004, 0.5))
    if discriminator_weights:
        discriminator.load_weights(discriminator_weights)
    discriminator.trainable = False
    # Create GAN
    generated_imgs = generator(img_input)
    gan = Model(img_input, [generated_imgs, discriminator(generated_imgs)])
    gan.compile(optimizer=Adam(0.0001, 0.5),
                loss=[perceptual_loss(input_shape=(*img_size, 3)), 'binary_crossentropy'],
                loss_weights=[10, 1])
    if gan_weights:
        gan.load_weights(gan_weights)
    discriminator.trainable = True

    return gan, generator, discriminator


def smooth_positive_labels(shape):
    return 0.9 + np.random.random(shape) / 10


def create_parallel_queue(data_seq):
    enqueuer = OrderedEnqueuer(
        data_seq,
        use_multiprocessing=True,
        shuffle=True)
    enqueuer.start(workers=4, max_queue_size=8)
    return enqueuer.get()


def train_discriminator(gan,
                        generator,
                        discriminator,
                        gray_imgs,
                        rgb_imgs,
                        batch_size=32):
    batch_y_shape = (len(gray_imgs), 1)
    colorized_images = generator.predict(gray_imgs)
    discriminator.trainable = True

    d_loss_real = discriminator.train_on_batch(rgb_imgs, smooth_positive_labels(batch_y_shape))
    d_loss_fake = discriminator.train_on_batch(colorized_images, np.zeros(batch_y_shape))
    return (d_loss_real + d_loss_fake) / 2.


def train_generator(gan,
                    discriminator,
                    gray_imgs,
                    rgb_imgs,
                    batch_size=32):
    discriminator.trainable = False
    targets = np.ones((batch_size // 2, 1))
    return gan.train_on_batch(gray_imgs, [rgb_imgs, targets])


def train(args, experiment=None):
    LOG_PERIOD = 1000
    # Dataset
    train_seq = TrainDatasetSequence(
        args.train_dataset,
        batch_size=args.batch_size // 2,
        img_size=(args.img_w, args.img_h))

    # Build GAN
    gan, generator, discriminator = build_gan(
        (args.img_w, args.img_h),
        args.generator_weights,
        args.discriminator_weights,
        args.gan_weights)

    # Log images callback
    on_batch_end = LogImages(
        experiment,
        paths=file_listing(args.validation_path),
        model=generator,
        img_size=(args.img_w, args.img_h),
        log_iters=LOG_PERIOD).on_batch_end

    # Create parallel queue to load images asynchronously
    batch_queue = create_parallel_queue(train_seq)

    # Train GAN
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}')

        for batch_idx in tqdm(range(len(train_seq))):
            gray_imgs, rgb_imgs = next(batch_queue)

            disc_loss = train_discriminator(
                gan,
                generator,
                discriminator,
                gray_imgs,
                rgb_imgs,
                args.batch_size)
            _, perceptual_loss, gen_loss = train_generator(
                gan,
                discriminator,
                gray_imgs,
                rgb_imgs,
                args.batch_size)
            # CometML logs
            metrics = {
                'discriminator_loss': disc_loss,
                'generator_loss':     gen_loss,
                'perceptual_loss':    perceptual_loss
            }
            experiment.log_metrics(metrics, step=batch_idx, epoch=epoch)
            on_batch_end(batch_idx, None)
            # Save model
            if batch_idx % LOG_PERIOD == 0:
                step = (batch_idx + epoch * len(train_seq)) / 1000
                save_path = f'{args.model_save_path}/gan_{args.img_w}x{args.img_h}_epoch-{epoch}_{step}K.h5'
                gan.save_weights(save_path)


if __name__ == '__main__':
    # Command line arguments parsing
    parser = argparse.ArgumentParser(description='Train a colorization deep learning model')
    parser.add_argument('--train-dataset',
                        type=str,
                        default=relative_path('../data/train/'),
                        help='Train dataset base path. Folder should contain subfolder for each class.')
    parser.add_argument('--validation-path',
                        type=str,
                        default=relative_path('../data/val/'),
                        help='Path to directory with validation images that will be uploaded to comet after each epoch')
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
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs')
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
