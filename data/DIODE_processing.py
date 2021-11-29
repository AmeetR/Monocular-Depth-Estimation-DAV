"""
DIODE data is organized hierarchically: 

Depth: 
- train/val
    - indoors
    - outdoors
        - scene<scene_number>
            - scan<scan_number>
                - <scene_id>_<scan_id>_<outdoor/indoor>_<image_name>.png
                - <scene_id>_<scan_id>_<outdoor/indoor>_<image_name>_depth.npy
                - <scene_id>_<scan_id>_<outdoor/indoor>_<image_name>_depth_mask.npy


Creating a pytorch data loader straight from this format will not be tremendously convenient. 
Let's first generate a CSV file that contains the paths to all the images, depth maps, and depth masks. 
It will also contain scene and scan information, as well as indoor and outdoor information.

"""
import os
import pandas as pd
from collections import defaultdict

class DIODE_processing:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.train_dir = os.path.join(self.root_dir, 'train')
        self.val_dir = os.path.join(self.root_dir, 'val')
        

    def generate_csv(self, dir, csv_file_name = 'train.csv'):
        """
        Generates CSV for train and val dirs. 
        """
        out_dict = defaultdict(list)

        for location in os.listdir(dir):
            current_path = os.path.join(dir, location)
            for scene in os.listdir(current_path):
                
                scene_path = os.path.join(current_path, scene)
                
                for scan in os.listdir(scene_path):
                    scan_path = os.path.join(scene_path, scan)
                    if os.path.isdir(scan_path):
                        for data in os.listdir(scan_path):
                            if data.endswith('.png'):
                                
                                img_path = os.path.join(scan_path, data)
                                out_dict['location'].append(location)
                                out_dict['scene'].append(scene)
                                out_dict['scan'].append(scan)
                                out_dict['img_path'].append(img_path)
                                depth_path = img_path.replace('.png', '_depth.npy')
                                out_dict['depth_path'].append(depth_path)
                                depth_mask_path = img_path.replace('.png', '_depth_mask.npy')
                                out_dict['depth_mask_path'].append(depth_mask_path)

        df = pd.DataFrame.from_dict(out_dict)
        csv_file_name = os.path.join(self.root_dir, csv_file_name)
        df.to_csv(csv_file_name, index=False)
        print(f'CSV file {csv_file_name} generated.')


if __name__ == '__main__':
    root_dir = 'C:\\Data\\DIODE'
    DIODE = DIODE_processing(root_dir)

