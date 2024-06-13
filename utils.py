from sklearn.decomposition import PCA
import numpy as np
import cv2
import torchvision.transforms as T
import geomloss

patch_h = 28
patch_w = 28
feat_dim = 1024

def calc_2d_bbox(xs, ys, im_size):
    bbTL = (max(xs.min() - 1, 0),
            max(ys.min() - 1, 0))
    bbBR = (min(xs.max() + 1, im_size[0] - 1),
            min(ys.max() + 1, im_size[1] - 1))
    return [bbTL[0], bbTL[1], bbBR[0] - bbTL[0], bbBR[1] - bbTL[1]]

def create_bounding_box(img, pose, pt_cld_data, intrinsic_matrix,color=(0,0,255)):
    "Create a bounding box around the object"
    # 8 corner points of the ptcld data
    line_with = 2
    min_x, min_y, min_z = np.min(pt_cld_data['x']), np.min(pt_cld_data['y']),np.min(pt_cld_data['z'])
    max_x, max_y, max_z = np.max(pt_cld_data['x']), np.max(pt_cld_data['y']),np.max(pt_cld_data['z'])

    corners_3D = np.array([[max_x, min_y, min_z],
                           [max_x, min_y, max_z],
                           [min_x, min_y, max_z],
                           [min_x, min_y, min_z],
                           [max_x, max_y, min_z],
                           [max_x, max_y, max_z],
                           [min_x, max_y, max_z],
                           [min_x, max_y, min_z]])

    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)

    # Perspective Projection to obtain 2D coordinates for masks
    homogenous_2D = intrinsic_matrix @ (pose @ homogenous_coordinate.T)
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)

    # Draw lines between these 8 points
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[4]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[2]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[5]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[3]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[6]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[3]), tuple(coord_2D[7]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[7]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[5]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[5]), tuple(coord_2D[6]), color, line_with)
    img = cv2.line(img, tuple(coord_2D[6]), tuple(coord_2D[7]), color, line_with)

    return img

def show_fature(features):

    features = features.reshape(patch_h * patch_w, feat_dim).cpu()

    # 创建PCA对象并拟合特征
    pca = PCA(n_components=3)
    pca.fit(features)

    # 对PCA转换后的特征进行归一化处理
    pca_features = pca.transform(features)
    for i in range(3):
        pca_features[:, i] = (pca_features[:, i] - pca_features[:, i].min()) \
                                     / (pca_features[:, i].max() - pca_features[:, i].min())
    pca_features_rgb = pca_features.reshape(patch_h, patch_w, 3)
    return pca_features_rgb

def read_txt_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            line = line.strip()
            data.append(int(line))
    return data


p = 2
entreg = .1 # entropy regularization factor for Sinkhorn
OTLoss = geomloss.SamplesLoss(
    loss='sinkhorn', p=p,
    # 对于p=1或p=2的情形
    cost=geomloss.utils.distances if p==1 else geomloss.utils.squared_distances,
    blur=entreg**(1/p), backend='tensorized')

# 定义图像转换操作
transform = T.Compose([
    T.Resize((patch_h * 14, patch_w * 14)),  # 调整图像大小
    T.ToTensor(),  # 转换为张量
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
])