import json
import os

OUTPUT_DIR = "/mnt/DATA/IMANE/p3_NY_full/data/224/annotations/blocks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for split in ['train', 'val', 'test']:
    in_file = f'annotations_all_{split}.json'
    out_file = os.path.join(OUTPUT_DIR, f'annotations_NY_{split}.json')
    print(f"Processing {split}...")
    with open(in_file, 'r') as f:
        data = json.load(f)
    ny_images = [img for img in data['images'] if '/NY/' in img['file_name']]
    ny_ids = {img['id'] for img in ny_images}
    ny_annotations = [ann for ann in data['annotations'] if ann['image_id'] in ny_ids]
    data['images'] = ny_images
    data['annotations'] = ny_annotations
    with open(out_file, 'w') as f:
        json.dump(data, f)
    print(f"  -> {len(ny_images)} NY images, {len(ny_annotations)} annotations")
