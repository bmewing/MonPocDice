import utils
import argparse
import glob
import os
import cv2

parser = argparse.ArgumentParser(description='Detect Monsterpocalypse Dice in my Dice Tray')
parser.add_argument('-i', '--input', dest='input_file', help='What file should be detected?',
                    required=False, default='images/full.jpg')
parser.add_argument('-d', '--dir', dest='input_dir', help='What directory should be processed?',
                    required=False)
parser.add_argument('-o', '--output', dest='output_dir', help='Where should results be stored?',
                    required=False)
parser.add_argument('-v', '--viz', dest='viz_clusters', action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    if args.input_dir is not None:
        files = glob.glob(args.input_dir+'*')
    else:
        files = [args.input_file]
    for f in files:
        image = utils.load_image(f)
        cropped = utils.crop_to_tray(image)
        dice, img = utils.detect_dice(cropped, viz_clusters=args.viz_clusters)

        if args.output_dir is not None:
            fp = args.output_dir+os.path.basename(f)
            cv2.imwrite(fp, img)
        else:
            utils.display_image(img)
