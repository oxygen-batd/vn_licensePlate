from paddleocr import PaddleOCR
import cv2
import numpy as np

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det =True,
    det_algorithm='DB',
    rec_algorithm='SVTR_LCNet')

def image_preprocessing(image):
    w,h = image.shape[:2]
    if h < 80:
        image = cv2.resize(image, (300, 300), fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    
    # PaddleOCR requires RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

def read_license_plate(license_plate_crop):
    try:
        image = image_preprocessing(license_plate_crop)
        results = ocr.ocr(image, cls=True)
        
        all_texts = []
        for res in results:
            for line in res:
                box = [tuple(point) for point in line[0]]
                # Finding the bounding box
                box = [(min(point[0] for point in box), min(point[1] for point in box)),
                       (max(point[0] for point in box), max(point[1] for point in box))]
                txt = line[1][0]
                all_texts.append(txt)
        
        if all_texts:
            license_plate_text = " ".join(all_texts)
        else:
            license_plate_text = None
        
        return license_plate_text
    
    except Exception as e:
        print(f"Error in reading license plate: {e}")
        return None

def limit_text_within_image(image, text, org, font, font_scale, thickness):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size

    org_x, org_y = org
    if org_x < 0:
        org_x = 0
    if org_y < text_height:
        org_y = text_height
    if org_x + text_width > image.shape[1]:
        org_x = image.shape[1] - text_width - 5
    if org_y - text_height < 0:
        org_y = text_height

    return (org_x, org_y)
