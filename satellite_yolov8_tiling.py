import numpy as np
import cv2

import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from sahi.predict import get_prediction, get_sliced_prediction
import cv2
import numpy as np
import time


yolo_model_path, yolo_version = ("runs/detect/Dotav2Dataset_yolov8x_550/weights/best.pt", "yolov8")

ADVANCE = 1
DEBUG = 1

detection_model = AutoDetectionModel.from_pretrained(
    model_type= yolo_version,
    model_path=yolo_model_path,
    confidence_threshold=0.5,
    device="cpu", # or 'cuda:0'
)


def start_yolo(image):
        
    try:     
        processed_image = image
        image = processed_image.copy()

        print("IMG Shape:", image.shape)

        if not ADVANCE:
            result = get_prediction(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detection_model)
        else:

            window_size = 1280 #Less this number, more number of divided images

            result = get_sliced_prediction(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                detection_model,
                slice_height = window_size,#image.shape[0]//4,#256,
                slice_width = window_size,#image.shape[1]//2,#256,
                overlap_height_ratio = 0.2,
                overlap_width_ratio = 0.2,
                verbose=DEBUG
            )

        yolo_res = result.to_coco_annotations()

        print( "Objects Detected:", len(yolo_res))
        img = np.array(result.image)
        
        all_names = []
        for detection in yolo_res:
            bbox = detection['bbox']
            conf = detection['score']
            cls = detection['category_id']
            name = detection['category_name']
            area = detection['area']
            iscrowd = detection['iscrowd']

            all_names.append(name)


            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))

            color = (100,100,0)
            thickness = 2
            img = cv2.rectangle(img, start_point, end_point, color, thickness)
            cv2.putText(img, str(name)+" "+str(conf)[:4],start_point, cv2.FONT_HERSHEY_PLAIN, 1, (50,50,200), 2)

        
        result.export_visuals(export_dir="./")

        cv2.imshow("Demo", img)
        print("#"*10)

        cv2.waitKey(10000)

    except Exception as e:
        print("ERROR:", e)

    finally:
        print("Releasing")
    

if __name__ == "__main__":
    image = cv2.imread("test_images/P0065.png")
    start_yolo(image)


