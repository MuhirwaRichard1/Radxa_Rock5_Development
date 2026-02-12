from ultralytics import YOLO

model = YOLO("./yolo26s_rknn_model")

# 0 = default webcam
results = model.predict(source=0, show=True)
