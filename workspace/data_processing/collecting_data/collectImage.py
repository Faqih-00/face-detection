import os
import uuid
import cv2

# ToDo:
# define path for train, test, and val
# capture image with opencv and save it to each path
# annotate our image with labelme

train_path = os.path.abspath(os.path.join('data', 'plain', 'train', 'images'))
test_path = os.path.abspath(os.path.join('data', 'plain', 'test', 'images'))
val_path = os.path.abspath(os.path.join('data', 'plain', 'val', 'images'))
number_images = 20
train_len = round(number_images * .7)
test_len = round(number_images * .15)
val_len = round(number_images * .15)
counter = 0
train_len_count = 0
test_len_count = 0
val_len_count = 0

cap = cv2.VideoCapture(1)
while counter < number_images:
    ret, frame = cap.read()
    cv2.imshow('image collection', frame)
    if cv2.waitKey(1) & 0XFF == ord('t'):
        if train_len_count < train_len:
            print(f'Collecting for train image {train_len_count+1}')
            imgname = os.path.join(train_path, f'{str(uuid.uuid1())}.jpg')
            cv2.imwrite(imgname, frame)
            train_len_count += 1
        elif train_len_count == train_len and test_len_count < test_len:
            print(f'Collecting for test image {test_len_count+1}')
            imgname = os.path.join(test_path, f'{str(uuid.uuid1())}.jpg')
            cv2.imwrite(imgname, frame)
            test_len_count += 1
        elif test_len_count == test_len and val_len_count < val_len:
            print(f'Collecting for validation image {val_len_count+1}')
            imgname = os.path.join(val_path, f'{str(uuid.uuid1())}.jpg')
            cv2.imwrite(imgname, frame)
            val_len_count += 1
        counter += 1
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
