import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='/root/data/',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--video_path',
        default='jpg',
        type=str,
        help='Directory path of Videos jpg')
    parser.add_argument(
        '--annotation_path',
        default='annotaion.json',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--label_path',
        default='label.csv',
        type=str,
        help='label file path')
    parser.add_argument(
        '--log_path',
        default='log',
        type=str,
        help='Log directory path')
    parser.add_argument(
        '--pretrained_model_path',
        default='',
        type=str,
        help='pre-train model directory path')
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--device',
        default="",
        type=str,
        help='gpu number you use')
    parser.add_argument(
        '--is_parallel',
        default=False,
        type=bool,
        help='')
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='Batch Size')

    args = parser.parse_args()
    return args
