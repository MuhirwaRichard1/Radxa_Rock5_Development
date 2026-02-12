from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26m.pt")

# Export the model to RKNN format
# 'name' can be one of rk3588, rk3576, rk3566, rk3568, rk3562, rv1103, rv1106, rv1103b, rv1106b, rk2118, rv1126b
model.export(format="rknn", name="rk3588")  # creates '/yolo26n_rknn_model'
