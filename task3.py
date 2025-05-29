import os
import mmcv
from mmengine.config import Config
from mmdet.apis import init_detector,inference_detector
from mmengine.visualization import Visualizer
from mmengine.registry import VISUALIZERS

'''config_file='configs/mask_rcnn_final.py'
checkpoint_file='work_dirs_mask_rcnn/mask_rcnn/epoch_10.pth'''

config_file='configs/sparse_rcnn_final.py'
checkpoint_file='work_dirs/sparse_rcnn/epoch_10.pth'

model=init_detector(config_file,checkpoint_file,device='cpu')

cfg=Config.fromfile(config_file)

visualizer:Visualizer=VISUALIZERS.build(cfg.visualizer)
visualizer.dataset_meta=model.dataset_meta

img_dir='imgs_task3'
img_list=sorted(os.listdir(img_dir))
#output_dir='outputs/task3_mask_rcnn'
output_dir='outputs/task3_sparse_rcnn'
os.makedirs(output_dir,exist_ok=True)






for img_name in img_list:
    img_path=os.path.join(img_dir,img_name)
    img=mmcv.imread(img_path)
    result=inference_detector(model,img)
    #out_file=os.path.join('outputs/task3_mask_rcnn',f'{img_name}_maskrcnn.jpg')
    out_file=os.path.join('outputs/task3_sparse_rcnn',f'{img_name}_sparsercnn.jpg')

    visualizer.add_datasample(
        name=img_name,
        image=img,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        pred_score_thr=0.3,
        out_file=out_file
    )
    print(f'Saved result to {out_file}')