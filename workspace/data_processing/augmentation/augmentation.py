import albumentations as alb
import cv2
import os
import json
import numpy as np

# ToDo:
# create augmentation pipeline
# create folder in parent for save our augmented data
# augmenting our data(images, labels) and save it to augmented data folder

augmentor = alb.Compose([
    alb.RandomCrop(width=480, height=480),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RandomFog(p=0.2),
    alb.VerticalFlip(p=0.2)
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

for partition in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join('data', 'plain', partition, 'images')):
        img = cv2.imread(os.path.join(
            'data', 'plain', partition, 'images', image))
        coords = [0, 0, 0.00001, 0.00001]
        label_path = os.path.join(
            'data', 'plain', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640, 480, 640, 480]))
        try:
            for x in range(60):
                augmented = augmentor(
                    image=img, bboxes=[coords], class_labels=['Face'])
                cv2.imwrite(os.path.join('data', 'augmented', partition, 'images',
                            f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0, 0, 0, 0]
                    annotation['class'] = 0

                with open(os.path.join('data', 'augmented', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)
