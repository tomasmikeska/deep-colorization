import os
import argparse
from comet_ml import Experiment
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.optimizers import Adam
from models.fusion_unet_v2 import FusionUNet
from losses.perceptual_loss import perceptual_loss
from dataset import TrainDatasetSequence, TestDatasetSequence
from callbacks.log_images import LogImages
from utils import relative_path, file_listing


def train(model, args, experiment=None):
    train_seq = TrainDatasetSequence(args.train_dataset,
                                     batch_size=args.batch_size,
                                     img_size=(args.img_w, args.img_h))
    test_seq = TestDatasetSequence(args.test_dataset,
                                   batch_size=args.batch_size,
                                   img_size=(args.img_w, args.img_h))
    model.compile(optimizer=Adam(lr=0.0002),
                  loss=perceptual_loss(input_shape=(args.img_h, args.img_w, 3)))
    model.summary()

    if args.weights:
        model.load_weights(args.weights)

    callbacks = [
        ModelCheckpoint(
            args.model_save_path + 'fusion_unet_{epoch:02d}_{val_loss:.3f}.h5',
            save_weights_only=True,
            verbose=1),
        TerminateOnNaN()
    ]
    if experiment is not None:
        callbacks.append(LogImages(experiment,
                                   paths=file_listing(args.validation_path),
                                   img_size=(args.img_w, args.img_h)))

    model.fit_generator(
        train_seq,
        epochs=args.epochs,
        validation_data=test_seq,
        use_multiprocessing=True,
        workers=8,
        callbacks=callbacks)


if __name__ == '__main__':
    # Command line arguments parsing
    parser = argparse.ArgumentParser(description='Train a colorization deep learning model')
    parser.add_argument('--train-dataset',
                        type=str,
                        default=relative_path('../data/imagenet-sample/train/'),
                        help='Train dataset base path. Folder should contain subfolder for each class.')
    parser.add_argument('--test-dataset',
                        type=str,
                        default=relative_path('../data/imagenet-sample/test/'),
                        help='Test dataset base path. Folder should contain images directly.')
    parser.add_argument('--validation-path',
                        type=str,
                        default=relative_path('../data/imagenet-sample/val/'),
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
    parser.add_argument('--weights',
                        type=str,
                        help='Model weights')
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
    model = FusionUNet((args.img_h, args.img_w, 3))
    train(model, args, experiment)
