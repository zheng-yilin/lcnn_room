import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import skimage.io as io
import scipy.io as scio

from search_polygon_predict import seg_color_trans, edge_embedding
from Hungarian_algorithm import match_trans_by_Hungarian

'''
indoorSeg方法：Indoor Scene Layout Estimation froma Single Image
代码作用：输出indoorSeg方法的效果图（包含原图，分割图，edge比较图），其中网络效果图是从黄仕中那里拷过来的

读出来的分割图是3通道的，(h, w, 3)=(320, 320, 3)
标签与像素值的对应关系：
前墙：1--(247,  0,  33)
左墙：2--(255， 219， 138)
右墙：3--(103， 188， 154)
地面：4--(16， 16， 69)
天花板：5--(236， 244， 193)
'''
# 1 wall                                                 # seed   lcnn    roomnet
# iname = 'sun_bbrgcasvzdvryynk'                                 #116     115
# iname = '1d55e3914a87c8c4f5d911c70ac82e31474f1d7f'             #117     121

# 2wall
# iname = '01fe1b8ceed678b27fd45aca2bde6b8c7f0cad40'             #124     126
# iname = '82977512d3b886eadf97fc3325ce2ed830c05e1b'             #140     139
# iname = 'sun_aawlhrzlsdywldmq'                                 #143     146
# iname = 'sun_acddebisonjmjrum'
# iname = 'sun_afywcuzscjctbdcu'
# iname = '0aa0b07b9fc1d7f59c55b585be3ee3a3e77f31fc'

# 3wall
# iname = '2c4a7c9e03e1d61ab2975a6df13b39b6ff6f4a6e'              #154    163
# iname = '4c07ebb141b726243b7dd9346054b693eeeea22a'              #166
# iname = '7e072d7ebc2cb048db52e25ea20322800ff8cf93'                # 1002  1003
# iname = 'sun_aclmoxwmazctsgls'                                    # 1010  1011
iname = 'sun_aiqzgxphqrsranga'                                      # 1012  1015

eval_order = 4 # 黄仕中给的第几个版本的预测效果图

srce_dataset_dir = '/home/hui/database/dataset/LSUN'
split = 'validation'
split_path = os.path.join(srce_dataset_dir, split)

pred_seg_path = os.path.join(srce_dataset_dir, 'huang_lsun', f'eval_{eval_order}',iname + '.png')
pred_seg = io.imread(pred_seg_path)
h, w, c = pred_seg.shape

gt_seg_path = os.path.join(split_path, 'layout_seg', iname + '.png')
gt_seg = io.imread(gt_seg_path)     # 是单通道的，但标签值不对(1-51, 2-102, 3-153, 4-204, 5, 255)
gt_seg = cv2.resize(gt_seg, (w, h))


image_path = os.path.join(split_path, 'images', iname + '.jpg')
img = io.imread(image_path)
img = cv2.resize(img, (w, h))

# 两个墙面的和三个墙面的还不一样
trans_dict = {247:1, 255:2, 103:3, 16:4, 236:5}
# trans_dict = {247:1, 255:2, 103:3, 16:4, 236:5}
# trans_dict = {255:2, 92:3, 0:4, 234:5}

# 先从3通道的颜色标签转换为(1,2,3,4,5)的标签
seg_layer1 = pred_seg[:, :, 0].copy()

#
for i in trans_dict.keys():
    mask_ = seg_layer1==i
    seg_layer1[mask_]=trans_dict[i]


pred_seg_trans = seg_color_trans((seg_layer1), 'Lsun')

gt_edge_embedding = edge_embedding(img, gt_seg, 'red')

gt_pred_edge_embedding = edge_embedding(gt_edge_embedding, seg_layer1, 'green')


# plt.figure(figsize=(12, 12))
# plt.imshow(gt_pred_edge_embedding)
# plt.show()


plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(pred_seg_trans)
plt.subplot(133)
plt.imshow(gt_pred_edge_embedding)
plt.show()

save_dir = os.path.join('/home/hui/database/dataset/LCNN_Data/Experiment_LCNN/Experiment_on_LSUN', 'seg_lsun', f'{iname}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img_save_path = os.path.join(save_dir, f'image_{iname}.png')
seg_save_path = os.path.join(save_dir, f'seg_{iname}.png')
edge_save_path = os.path.join(save_dir, f'edge_{iname}.png')
path_list = [img_save_path, seg_save_path, edge_save_path]

data_list = [img, pred_seg_trans, gt_pred_edge_embedding]

for data, path in zip(data_list, path_list):
    plt.figure(figsize=(5, 5))  # figsize中高宽的比例，要和待保存图片的高宽比例一致
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.imshow(data)
    plt.savefig(path)
    # plt.show()








