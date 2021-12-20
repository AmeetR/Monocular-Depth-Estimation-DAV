import open3d
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_depth_map(dm, validity_mask):
    validity_mask = validity_mask > 0
    MIN_DEPTH = 0.
    MAX_DEPTH = min(300, np.percentile(dm, 99))
    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    dm = np.log(dm, where=validity_mask)

    dm = np.ma.masked_where(~validity_mask, dm)

    cmap = plt.cm.jet
    cmap.set_bad(color='black')
    plt.imshow(dm, cmap=cmap, vmax=np.log(MAX_DEPTH))

intrinsic = open3d.camera.PinholeCameraIntrinsic()
[width, height, fx, fy, cx, cy] = [1024, 768, 886.81, 927.06, 512, 384]
intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
idx = 255

data = pd.read_csv('C:\\Data\\DIODE\\train.csv')
example = data['img_path'][idx]
example_depth = (np.load(data['depth_path'][idx]) * 1000).astype('uint16')
cv2.imwrite(data['depth_path'][idx].replace('.npy', '.png'), example_depth)
color_raw = open3d.io.read_image(example)
depth_raw = open3d.io.read_image(data['depth_path'][idx].replace('.npy', '.png'))
rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
    color_raw, depth_raw)

validity_mask = np.load(data['depth_mask_path'][idx])



point_cloud = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
#point_cloud = open3d.geometry.PointCloud.create_from_depth_image(depth_raw , intrinsic, depth_scale=1.0)
print(point_cloud)
downpcd = point_cloud.voxel_down_sample(voxel_size=0.05)
open3d.visualization.draw_geometries([downpcd])
plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
depth_array =  np.asarray(rgbd_image.depth)
MAX_DEPTH = min(300, np.percentile(depth_array, 99))
plot_depth_map(depth_array, validity_mask)

plt.show()
