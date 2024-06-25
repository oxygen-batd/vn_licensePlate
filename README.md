Bài toán xử lý:
 - Task: Object Detection
 - Models: YOLOv8n

Step thực hiện:

1. Tạo dataset với Roboflow
	- Train/Test Split
	- Preprocessing
	  + Auto-Orient
	  + Resize
	  + ... 
	- Augmentation
	  + Flip
	  + Blur
	  + ... 
	- Export dataset
Link ví dụ: https://app.roboflow.com/o23ps6a/license_plates_vietnam/deploy

2. Train với YOLOv8  sử dụng dataset được export ở bước 1.
   Link ví dụ: https://colab.research.google.com/drive/1V-PBp6GONFWKkotKdQdWKuC7QUcr8J5H

3. Export model và sử dụng

![image](https://github.com/oxygen-batd/vn_licensePlate/assets/167840668/87d5640a-e3ec-4b0c-a9f1-c4579a87e0c9)

4. Các thuộc tính quan trọng khi thực hiện task **predict** : https://docs.ultralytics.com/modes/predict/#working-with-results

Tài liệu tham khảo:

- Ultralytics YOLO Docs: https://docs.ultralytics.com/tasks/detect/
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/quickstart_en.md
