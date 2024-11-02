import h5py

weights_path = 'C:\\Users\\danso\\Documents\\IA-projet\\ObjectDetection\\CarPlateDetection\\CarPlate_MaskRCNN\\logs\\custom20241101T2253\\mask_rcnn_custom_0001.h5'
with h5py.File(weights_path, 'r') as f:
    print(list(f.keys()))  