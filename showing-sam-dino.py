import argparse
import random
import time
from pathlib import Path
import numpy as np

import torch
from PIL import Image
import os
import torchvision
from torchvision.ops.boxes import batched_nms
import cv2
from util.slconfig import DictAction,SLConfig
import json
import util.misc as utils

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                             'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='/media/mldadmin/home/s125mdg35_05/DINO/DINO/image_out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/media/mldadmin/home/s125mdg35_05/DINO/DINO/logs/DINO/R50-MS4/checkpoint_best_regular.pth', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default="True")
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors

def box_cxcywh_to_xyxy(x):
    # 将DETR的检测框坐标(x_center,y_cengter,w,h)转化成coco数据集的检测框坐标(x0,y0,x1,y1)
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    # 把比例坐标乘以图像的宽和高，变成真实坐标
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # 把检测框画到图片上
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, tl, cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

path="/media/mldadmin/home/s125mdg35_05/dataset/data/annotations/instances_val2017.json"
with open(path, 'r') as f:
    dict = json.load(f)
print(dict.keys())
cata=[]
CATA=dict['categories']
print(len(CATA))
for i in range(0,len(CATA)):
    A=CATA[i]
    cata.append(A['name'])
CLASSES =cata

def filter_boxes(scores, boxes, confidence=0.5, apply_nms=True, iou=0.5):
    # 筛选出真正的置信度高的框
    keep = scores.max(-1).values > confidence
    scores, boxes = scores[keep], boxes[keep]

    if apply_nms:
        top_scores, labels = scores.max(-1)
        keep = batched_nms(boxes, top_scores, labels, iou)
        scores, boxes = scores[keep], boxes[keep]

    return scores, boxes

def main(args,compare=1):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(args)
    device = "cuda"

    # ------------------------------------导入网络
    # 下面的criterion是算损失函数要用的，推理用不到,postprocessors是解码用的，这里也没有用，用的是自己的。
    model, criterion, postprocessors = build_model_main(args)

    # ------------------------------------加载权重
    checkpoint = torch.load(args.resume, map_location='cuda')
    if "label_enc.weight" in checkpoint['model']:
        del checkpoint['model']["label_enc.weight"]
    if "label_enc.bias" in checkpoint['model']:
        del checkpoint['model']["label_enc.bias"]
    model.load_state_dict(checkpoint['model'], strict=False)

    # ------------------------------------把权重加载到gpu或cpu上
    model.to(device)

    # ------------------------------------打印出网络的参数大小
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameters:", n_parameters)

    # ------------------------------------设置好存储输出结果的文件夹
    output_dir = Path(args.output_dir)

    # -----------------------------------读取数据集,进行推理
    image_Totensor = torchvision.transforms.ToTensor()
    image_file_path = os.listdir("/media/mldadmin/home/s125mdg35_05/dataset/data/val2017")
    image_set = []
    image_file_path=sorted(image_file_path)
    #for image_item in image_file_path:
    for i in range(17800,18000):
        image_item=image_file_path[i]
        print("inference_image:", image_item)
        image_path = os.path.join("/media/mldadmin/home/s125mdg35_05/dataset/data/val2017", image_item)
        image_orig=image = Image.open(image_path)
        if compare!=1:
            image_tensor = image_Totensor(image)  # 假设 image_Totensor 返回一个 PyTorch 张量
            image_tensor = image_tensor.unsqueeze(0).to(device)  # 添加 batch 维度并移动到设备

            # 推理
            start_time = time.time()
            model.eval()
            with torch.no_grad():  # 禁用梯度计算以节省内存
                inference_result = model(image_tensor)
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time:.4f} seconds")

            # 处理推理结果
            probas = inference_result['pred_logits'].softmax(-1)[0, :, :-1].cpu()  # 获取概率并移到 CPU
            bboxes_scaled = rescale_bboxes(inference_result['pred_boxes'][0,].cpu(),
                                           (image_tensor.shape[3], image_tensor.shape[2]))  # 缩放边界框
            scores, boxes = filter_boxes(probas, bboxes_scaled)  # 过滤边界框
            scores = scores.numpy()  # 转换为 NumPy 数组
            boxes = boxes.numpy()

            # 打印边界框信息
            print("Detected boxes:")
            print(boxes)

            # 在图像上绘制边界框和标签
            image_np = np.array(image)  # 将 PIL 图像转换为 NumPy 数组
            for i in range(boxes.shape[0]):
                class_id = scores[i].argmax()
                label = CLASSES[class_id - 1]  # 假设 CLASSES 是一个类别列表
                confidence = scores[i].max()
                text = f"{label} {confidence:.3f}"
                plot_one_box(boxes[i], image_np, label=text)  # 在图像上绘制边界框

            # 显示图像
            #cv2.imshow("Detected Objects", image_np)
            cv2.waitKey(1)

            # 保存结果图像
            output_image = Image.fromarray(image_np)  # 将 NumPy 数组转换回 PIL 图像
            output_image.save(os.path.join(args.output_dir, image_item))

            # 释放 GPU 缓存
            torch.cuda.empty_cache()

            # 关闭 OpenCV 窗口
            cv2.destroyAllWindows()

            print("Processing complete.")

        #标准原图片
        if compare==1:
            Images0=dict['images']
            for i in range(0,len(Images0)):
                if Images0[i]["file_name"]==image_item:
                    image_id=Images0[i]["id"]
                    break
            Annotations0 = dict['annotations']
            category_id=[]
            bbox=[]
            for j in range(0,len(Annotations0)):
                if Annotations0[j]['image_id']==image_id:
                    category_id.append(Annotations0[j]['category_id'])
                    bbox.append(Annotations0[j]['bbox'])
            print(bbox)
            for k in range(len(bbox)):
                x_c, y_c, w, h = bbox[k]
                bbox[k]=[x_c, y_c ,
                 (x_c + w), (y_c + h)]
            for i in range(len(bbox)):
                class_id = category_id[i]
                label = CLASSES[class_id-1]
                image_orig = np.array(image_orig)
                plot_one_box(bbox[i], image_orig, label=label)
            image_orig = np.array(image_orig)
            #cv2.imshow("images", image_orig)
            cv2.waitKey(1)
            image_orig = Image.fromarray(image_orig)
            output_path="/media/mldadmin/home/s125mdg35_05/DINO/DINO/image_out"
            image_orig.save(os.path.join(output_path, image_item))

        cv2.destroyAllWindows()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args,compare=0)
