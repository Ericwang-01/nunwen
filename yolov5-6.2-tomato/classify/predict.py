# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run classification inference on images

Usage:
    $ python classify/predict.py --weights yolov5s-cls.pt --source im.jpg
"""

import argparse
import os
import sys
from fileinput import filename
from pathlib import Path
import logging

import cv2
import torch.nn.functional as F
from dataread import MyData

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from classify.train import imshow_cls
from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.general import LOGGER, check_requirements, colorstr, increment_path, print_args
from utils.torch_utils import select_device, smart_inference_mode, time_sync


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-cls.pt',  # model.pt path(s)
        source=ROOT / 'data/images/bus.jpg',  # file/dir/URL/glob, 0 for webcam
        imgsz=224,  # inference size
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        show=True,
        project=ROOT / 'runs/predict-cls',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
):
    file = str(source)
    seen, dt = 1, [0.0, 0.0, 0.0]
    device = select_device(device)
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Transforms
    transforms = classify_transforms(imgsz)

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup

    # Image
    t1 = time_sync()
    im = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    im = transforms(im).unsqueeze(0).to(device)
    im = im.half() if model.fp16 else im.float()
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    results = model(im)
    t3 = time_sync()
    dt[1] += t3 - t2

    p = F.softmax(results, dim=1)  # probabilities
    i = p.argsort(1, descending=True)[:, :5].squeeze()  # top 5 indices
    dt[2] += time_sync() - t3
    LOGGER.info(f"image 1/1 {file}: {imgsz}x{imgsz} {', '.join(f'{model.names[j]} {p[0, j]:.2f}' for j in i)}")
    output=f"{file}"
    test1=f"{', '.join(f'{model.names[j]}' for j in i)}"
    # 使用逗号分割字符串
    fields = test1.split(', ')
    # 提取第一个字段
    first_field = fields[0]
    file_name = 'output.txt'
    if first_field == "Apple":
        # print(first_field)
        # 打开文件并写入内容
        # with open('output.txt', 'w') as file:
        #     file.write(output)
        file = open(file_name, 'a+')
        file.write(output + '\n')
        file.close()

    # print(test1)
    # file_name = 'output.txt'
    # # 打开文件并写入内容
    # # with open('output.txt', 'w') as file:
    # #     file.write(output)
    # file = open(file_name,'a+')
    # file.write(output+'\n')
    # file.close()


    # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # shape = (1, 3, imgsz, imgsz)
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}' % t)
    # if show:
    #     imshow_cls(im, f=save_dir / Path(file).name, verbose=True)
    # # LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    # return p


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'D:/YOLO/yolov5-6.2-tomato/runs/train-cls/exp8/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=r'', help='file')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/predict-cls', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    root_dir = "C:/Users/24686/Desktop/appletest/test/"#当前根目录，在当前文件夹就不写
    image = "Apple"#图片文件夹名字，需要跟detect.py同一个文件夹下面
    img = MyData(root_dir,image)
    print(len(img))
    print(img[20000])
    for i in range(0,len(img)):
        opt.source = img[i]
        main(opt)

