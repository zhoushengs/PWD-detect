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

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i','--image_dir',default='D:\\Guelph\\data\\sicktree4yolo\\sicktree4yolo\\train\\images', help='图片文件夹路径')
    p.add_argument('-o','--output_dir', default='D:\\Guelph\\data\\sicktree4yolo\\sicktree4yolo\\train', help='输出 txt 存放目录')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    split_dataset(args.image_dir, args.output_dir, seed=args.seed)
    #write_all_images('D:\\Guelph\\data\\sicktree4yolo\\sicktree4yolo\\val\\images', os.path.join('D:\\Guelph\\data\\sicktree4yolo\\sicktree4yolo\\val', 'all.txt'))