import os, random, argparse

def split_dataset(image_dir, output_dir, ratio=(8,2), seed=42):
    # ratio 分别对应 train:val:test 的权重
    imgs = [f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))]
    random.seed(seed)
    random.shuffle(imgs)
    total = len(imgs)
    s = sum(ratio)
    n_train = int(total * ratio[0] / s)
    #n_val   = int(total * ratio[1] / s)
    splits = {
        'train.txt': imgs[:n_train],
        'val.txt':   imgs[n_train:]
        #'test.txt':  imgs[n_train+n_val:]
    }
    os.makedirs(output_dir, exist_ok=True)
    for fn, lst in splits.items():
        with open(os.path.join(output_dir, fn), 'w') as f:
            for img in lst:
                f.write(f"/images/{img}\n")

def write_all_images(image_dir, output_file):
    # 列出所有图片文件
    imgs = [f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))]
    imgs.sort()  # 可选：排序
    # 写入 output_file
    print(f"Writing {len(imgs)} images to {output_file}")
    with open(output_file, 'w') as f:
        for img in imgs:
            f.write(f"/images/{img}\n")  # 或者 f"/images/{img}\n"

def check_bbox():
        from matplotlib import pyplot as plt
        img = batch['img'][0].detach().cpu()
    # [C,H,W] → [H,W,C]
        img = img.permute(1,2,0).numpy()
        img = img * 255
        img = img.astype('uint8')
        b = batch['bboxes'][0].detach().cpu()
        H, W = img.shape[:2]
        plt.figure(figsize=(6,6))
        plt.imshow(img)
        if b.ndim == 1:
            b = b.unsqueeze(0)
        for x_c,y_c,w,h in b:  # 默认 xywh
            x1 = (x_c - w/2) * W
            y1 = (y_c - h/2) * H
            rect = plt.Rectangle((x1,y1), w*W, h*H, fill=False, edgecolor='r', linewidth=2)
            plt.gca().add_patch(rect)
        plt.axis('off')
        plt.show()

import cv2
import torch
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward

def visualize_haar_wavelet_color(img_path: str):
    """
    对 RGB 图像做一次级联 1 级 Haar 小波分解，
    并把原图、平均后的低频子带 LL 及高频子带 HL、LH、HH 
    显示在同一行子图中。
    """
    # 1) 读取彩色图并转为 RGB 归一化 [0,1]
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype('float32') / 255.0  # H×W×3

    # 2) 转为 Tensor[B,C,H,W]
    x = torch.from_numpy(img_rgb.transpose(2,0,1)).unsqueeze(0)  # [1,3,H,W]

    # 3) 1 级 DWTForward（Haar）
    dwt = DWTForward(J=1, mode='zero', wave='haar')
    Yl, Yh = dwt(x)  
    # Yl: [1,3,H/2,W/2], Yh[0]: [1,3,3,H/2,W/2]

    # 4) 平均各通道得到灰度子带
    LL = Yl[0].mean(0).cpu().numpy()             # [H/2,W/2]
    subbands = Yh[0][0]                          # [3,3,H/2,W/2] -> actually [C,3,H/2,W/2]
    HL = subbands[:,0].mean(0).cpu().numpy()     # [H/2,W/2]
    LH = subbands[:,1].mean(0).cpu().numpy()
    HH = subbands[:,2].mean(0).cpu().numpy()

    # 5) 绘图
    titles = ['Original RGB', 'LL (Low)', 'HL (Hori)', 'LH (Vert)', 'HH (Diag)']
    maps   = [img_rgb,      LL,           HL,           LH,           HH]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    # 原图
    axes[0].imshow(maps[0])
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    # 子带
    for ax, m, t in zip(axes[1:], maps[1:], titles[1:]):
        ax.imshow(m, cmap='gray', vmin=m.min(), vmax=m.max())
        ax.set_title(t)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 例子：替换为你的图片路径  "E:\projects\pytorch\yolo11\datasets\coco8\images\train\000000000009.jpg"
    #'D:\\programming\\testproject\\yolov8\\ultralytics\\datasets\\sicktree\\images\\02104.jpg'
    visualize_haar_wavelet_color("E:\\projects\\pytorch\\yolo11\\datasets\\coco8\\images\\train\\000000000009.jpg")

# if __name__ == '__main__':
#     p = argparse.ArgumentParser()
#     p.add_argument('-i','--image_dir',default='D:\\Guelph\\data\\sicktree4yolo\\sicktree4yolo\\train\\images', help='图片文件夹路径')
#     p.add_argument('-o','--output_dir', default='D:\\Guelph\\data\\sicktree4yolo\\sicktree4yolo\\train', help='输出 txt 存放目录')
#     p.add_argument('--seed', type=int, default=42)
#     args = p.parse_args()
#     split_dataset(args.image_dir, args.output_dir, seed=args.seed)
#     #write_all_images('D:\\Guelph\\data\\sicktree4yolo\\sicktree4yolo\\val\\images', os.path.join('D:\\Guelph\\data\\sicktree4yolo\\sicktree4yolo\\val', 'all.txt'))