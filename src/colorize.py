import sys
import numpy as np
import argparse
import skimage
from keras.models import load_model
from models.att_unet import AttentionUNet
from dataset import read_image


def build_model(args):
    if args.weights:
        model = AttentionUNet((args.img_h, args.img_w, 3))
        model.load_weights(args.weights)
        return model
    elif args.model:
        return load_model(args.model)
    else:
        sys.exit('One of --model or --weights command line params must be specified')


def colorize(img_path, model, img_size):
    img_rgb  = read_image(img_path, size=img_size)
    img_gray = skimage.color.rgb2gray(img_rgb).reshape(img_size + (1,))
    img_gray = np.repeat(img_gray, 3, axis=-1)  # Repeat channel to keep input 3-dimensional
    pred     = model.predict(img_gray.reshape((1, *img_size, 3)))[0]
    return np.rint(pred * 127.5 + 127.5).astype(np.uint8)  # Rescale to (0,255)


if __name__ == '__main__':
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Colorize provided images using selected model')
    parser.add_argument('--model',
                        type=str,
                        help='Saved model path')
    parser.add_argument('--weights',
                        type=str,
                        help='Saved model weights - required when --model not specified. Uses fusion unet.')
    parser.add_argument('--source',
                        type=str,
                        help='Source image path, either grayscale or rgb')
    parser.add_argument('--output',
                        type=str,
                        default='output.jpeg',
                        help='Output image path')
    parser.add_argument('--img-w',
                        type=int,
                        default=256,
                        help='Input image width to use as model input dim')
    parser.add_argument('--img-h',
                        type=int,
                        default=256,
                        help='Input image height to use as model input dim')
    args = parser.parse_args()
    # Colorize and save
    model = build_model(args)
    img_np = colorize(args.source, model, img_size=(args.img_w, args.img_h))
    skimage.io.imsave(args.output, img_np)
