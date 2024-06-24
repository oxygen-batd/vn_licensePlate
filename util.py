from paddleocr import PaddleOCR
import cv2

# init paddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', det =False)

def read_license_plate(license_plate_crop):
    try:
        # Convert license plate crop to RGB (PaddleOCR requires RGB)
        license_plate_crop_rgb = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)
        
        # Perform OCR
        results = ocr.ocr(license_plate_crop_rgb, cls=True)
        
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
