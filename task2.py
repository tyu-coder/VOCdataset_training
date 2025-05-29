import os
import mmcv
from mmengine.config import Config
from mmdet.apis import init_detector,inference_detector
from mmdet.visualization import DetLocalVisualizer

'''config_file='configs/mask_rcnn_final.py'
checkpoint_file='work_dirs_mask_rcnn/mask_rcnn/epoch_10.pth'
img_dir='data/SBD/benchmark_RELEASE/dataset/img'
output_dir='outputs/task2_mask_rcnn'
os.makedirs(output_dir,exist_ok=True)'''

config_file='configs/sparse_rcnn_final.py'
checkpoint_file='work_dirs/sparse_rcnn/epoch_10.pth'
img_dir='data/SBD/benchmark_RELEASE/dataset/img'
output_dir='outputs/task2_sparse_rcnn'
os.makedirs(output_dir,exist_ok=True)

cfg=Config.fromfile(config_file)
model=init_detector(config_file,checkpoint_file,device='cpu')

visualizer=DetLocalVisualizer()
visualizer.dataset_meta=model.dataset_meta

img_list=sorted(os.listdir(img_dir))[:4]
img_paths=[os.path.join(img_dir,fname) for fname in img_list]

for idx,img_path in enumerate(img_paths):
    img=mmcv.imread(img_path)
    result=inference_detector(model,img)

    visualizer.add_datasample(
        name=f'result_{idx}',
        image=img,
        data_sample=result,
        draw_gt=False,
        draw_pred=True
    )
    vis_img=visualizer.get_image()
    out_path=os.path.join(output_dir,f'vis_result_{idx}.jpg')
    mmcv.imwrite(vis_img,out_path)
    print(f'Saved result to {out_path}')