from scipy.io import loadmat
import os.path as osp
import numpy as np
import argparse

def convert_meta(meta, output, margin=10, dimension=(1280, 720)):
    matrix = loadmat(meta)
    videos = matrix['video'][0]
    for video in videos:
        vid = video[6][0]
        for frame in vid:
            frame_no = frame[0].item()
            out_str = ''
            for i, poly in enumerate(frame):
                box = np.array(poly)
                if box.size > 2:
                    # x, ycoords
                    top_left = [np.min(box[:, 0]), np.min(box[:, 1])]
                    bottom_right = [np.max(box[:, 0]), np.max(box[:, 1])]

                    x_width = bottom_right[0] - top_left[0] + margin / 2
                    y_width = bottom_right[1] - top_left[1] + margin / 2
                    x_center = (top_left[0] + bottom_right[0]) / 2
                    y_center = (top_left[1] + bottom_right[1]) / 2

                    x_width = x_width / dimension[0]
                    y_width = y_width / dimension[1]
                    x_center = x_center / dimension[0]
                    y_center = y_center / dimension[1]

                    if i == 1 or i == 3:
                        label = 0
                    else:
                        label = 0
                    out_str = out_str + '%s %s %s %s %s \n' % (label, x_center, y_center, x_width, y_width)
                out_name = 'frame_%s.txt' % str(frame_no).zfill(4)
                with open(osp.join(output, out_name), 'w') as outfile:
                    outfile.write(out_str)

def parseargs():
    parser = argparse.ArgumentParser('get bboxes in yolo format')
    parser.add_argument('-i', help='metadata', default=None)
    parser.add_argument('-o', help='output annotation folder')
    return parser.parse_args()

def main(args):
    if args.i is not None:
        convert_meta(args.i, args.o)

if __name__ == '__main__':
    args = parseargs()
    main(args)
