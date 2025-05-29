import os
import json
import numpy as np
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as maskUtils

def load_mat_file(inst_path,cls_path):
    inst_data=loadmat(inst_path)['GTinst'][0][0]
    cls_data=loadmat(cls_path)['GTcls'][0][0]

    inst_mask=inst_data['Segmentation']
    cls_mask=cls_data['Segmentation']
    categories=cls_data['CategoriesPresent'].squeeze().tolist()

    if isinstance(categories,int):
        categories=[categories]
    return inst_mask,cls_mask,categories

def binary_mask_to_rle(binary_mask):
    rle=maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts']=rle['counts'].decode('utf-8')
    return rle

def create_coco_json(sbd_root,split_file,output_file):
    with open(split_file,'r') as f:
        image_ids=[line.strip() for line in f.readlines()]

    coco={
        "images":[],
        "annotations":[],
        "categories":[]
    }

    category_names=[
        "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
    ]
    for idx,name in enumerate(category_names):
        coco["categories"].append({
            "id":idx+1,
            "name":name,
            "supercategory":"object"
        })
    annotation_id=1
    for img_id,img_name in enumerate(tqdm(image_ids,desc="转换中")):
        jpg_file=os.path.join(sbd_root,'img',img_name+'.jpg')
        if not os.path.exists(jpg_file):
            print(f"跳过不存在的图片：{jpg_file}")
            continue
        with Image.open(jpg_file) as img:
            width,height=img.size
        coco["images"].append({
            "id":img_id,
            "file_name":img_name+".jpg",
            "width":width,
            "height":height
        })

        inst_path=os.path.join(sbd_root,'inst',img_name+'.mat')
        cls_path=os.path.join(sbd_root,'cls',img_name+'.mat')

        if not os.path.exists(inst_path) or not os.path.exists(cls_path):
            print(f"跳过缺失的标注：{img_name}")
            continue
        inst_mask,cls_mask,present_cats=load_mat_file(inst_path,cls_path)

        for inst_id in np.unique(inst_mask):
            if inst_id==0:
                continue

            binary_mask=(inst_mask==inst_id).astype(np.uint8)
            y_indices,x_indices=np.where(binary_mask)
            if y_indices.size==0 or x_indices.size==0:
                continue

            category_id=int(cls_mask[y_indices[0],x_indices[0]])
            rle=binary_mask_to_rle(binary_mask)
            area=int(maskUtils.area(rle))
            bbox=maskUtils.toBbox(rle).tolist()

            coco["annotations"].append({
                "id":annotation_id,
                "image_id":img_id,
                "category_id":category_id,
                "segmentation":rle,
                "area":area,
                "bbox":bbox,
                "iscrowd":0
            })
            annotation_id+=1
    with open(output_file,'w') as f:
        json.dump(coco,f)
    print(f"\n COCO JSON saved to {output_file}")

if __name__=="__main__":
    sbd_root=os.path.join("data","SBD","benchmark_RELEASE","dataset")
    train_txt=os.path.join(sbd_root,"train.txt")
    val_txt=os.path.join(sbd_root,"val.txt")
    create_coco_json(sbd_root,train_txt,"sbd_train_coco.json")
    create_coco_json(sbd_root,val_txt,"sbd_val_coco.json")


