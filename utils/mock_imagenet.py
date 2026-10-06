import os
from PIL import Image
import numpy as np

base_dir = "/local2/marcio/DOUTORADO/LINDA/runLINDA/case_studies/cds/cds_tier_3/LINDA_data"

# ImageNet structure: train/class_xxx/img.jpg
splits = ['train', 'val']
num_classes = 1000
imgs_per_class = 10

print(f"Creating Mock ImageNet at: {base_dir}")

for split in splits:
    for i in range(num_classes):
        class_name = f"n{str(i).zfill(8)}"
        class_dir = os.path.join(base_dir, split, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for j in range(imgs_per_class):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(os.path.join(class_dir, f"mock_{j}.jpg"))

print("Done! Try running NodeAgent again.")