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


def colorize(img_path, model, img_size, show_original=False):
    '''
    Colorize image with specified model. Since human eye is much more sensitive to luminance (lightness change)
    than to chrominance (color change), we convert image to specified image size needed by the model
    used (e.g. 192x192), colorize it and then assemble final image in CIE LAB color space by putting together
    original image lightness and colorized image's AB channels resized to original image size.

    Args:
        img_path (string): JPEG image full path.
        model (Keras model): Model instance to use.
        img_size (tuple): Model input image size. Output image will keep the size of original image.
        show_original (bool): Concatenate original image with colorized image
    '''
    orig_rgb   = read_image(img_path)
    orig_lab   = skimage.color.rgb2lab((orig_rgb + 1) / 2)
    input_rgb  = skimage.transform.resize(orig_rgb, img_size)
    input_gray = skimage.color.rgb2gray(input_rgb).reshape(img_size + (1,))
    input_gray = np.repeat(input_gray, 3, axis=-1)  # Repeat channel to keep input 3-dimensional
    output_rgb = model.predict(input_gray.reshape((1, *img_size, 3)))[0] / 2 + 0.5  # Colorize
    output_lab = skimage.color.rgb2lab(output_rgb)  # Convert colorized image to LAB
    output_lab = skimage.transform.resize(output_lab, (orig_rgb.shape[0], orig_rgb.shape[1]))  # Resize LAB to orig size
    final_lab  = np.zeros((orig_rgb.shape[0], orig_rgb.shape[1], 3))  # Finals image LAB image
    final_lab[:, :, 0] = orig_lab[:, :, 0]  # Original image lightness channel
    final_lab[:, :, 1:] = output_lab[:, :, 1:]  # Take colorized image AB channels
    final_rgb  = skimage.color.lab2rgb(final_lab)

    if show_original:
        final_rgb = np.concatenate(((orig_rgb + 1) / 2, final_rgb), axis=1)

    return np.rint(final_rgb * 255).astype(np.uint8)  # Rescale to (0,255)


if __name__ == '__main__':
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Colorize provided images using selected model')
    parser.add_argument('--model',
                        type=str,
                        help='Saved model path')
    parser.add_argument('--weights',
                        type=str,
                        help='Saved model weights - required when --model not specified. Uses Attention U-Net.')
    parser.add_argument('--source',
                        type=str,
                        help='Source image path, either grayscale or rgb')
    parser.add_argument('--output',
                        type=str,
                        default='output.jpg',
                        help='Output image path')
    parser.add_argument('--img-w',
                        type=int,
                        default=192,
                        help='Input image width to use as model input dim')
    parser.add_argument('--img-h',
                        type=int,
                        default=192,
                        help='Input image height to use as model input dim')
    parser.add_argument('--show-original',
                        type=bool,
                        default=False,
                        help='Concatenate original image with colorized image')
    args = parser.parse_args()
    # Colorize and save
    model = build_model(args)
    img_np = colorize(args.source, model, img_size=(args.img_w, args.img_h), show_original=args.show_original)
    skimage.io.imsave(args.output, img_np)
