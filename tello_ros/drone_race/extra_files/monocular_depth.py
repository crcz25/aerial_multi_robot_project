# import cv2
# import torch
# import urllib.request

# import matplotlib.pyplot as plt

# model_type = "MiDaS_small"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# midas = torch.hub.load("intel-isl/MiDaS", model_type)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform

# img = cv2.imread('Drone Image_screenshot_11.04.2023.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# input_batch = transform(img).to(device)

# with torch.no_grad():
#     prediction = midas(input_batch)

#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()

# output = prediction.cpu().numpy()

# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(img)
# ax[1].imshow(output, cmap="magma")
# plt.show()


import cv2
import numpy as np
from openvino.runtime import Core
import matplotlib.pyplot as plt
ie = Core()

devices = ie.available_devices

for device in devices:
    device_name = ie.get_property(device, "FULL_DEVICE_NAME")
    print(f"{device}: {device_name}")

classification_model_xml = "openvino_midas_v21_small_256.xml"
model = ie.read_model(model=classification_model_xml)
compiled_model = ie.compile_model(model=model, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
print("INPUT", input_layer.shape)
print(output_layer.shape)
image = cv2.imread('Drone Image_screenshot_11.04.2023.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# image = cv2.imread('Drone Image_screenshot_10.04.2023.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# image = cv2.imread('15.jpg', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Blur the image to reduce noise
image = cv2.GaussianBlur(image, (5, 5), 0)
# apply histogram equalization 
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)

# N,C,H,W = batch size, number of channels, height, width.
N, C, H, W = input_layer.shape
# OpenCV resize expects the destination size as (width, height).
resized_image = cv2.resize(src=image, dsize=(W, H))
print(resized_image.shape)

input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)
print(input_data.shape)

# for single input models only
# result = compiled_model(input_data)[output_layer]

# for multiple inputs in a list
# result = compiled_model([input_data])[output_layer]

# or using a dictionary, where the key is input tensor name or index
result = compiled_model({input_layer.any_name: input_data})[output_layer]
print(result[0].shape)
print('result', result[0].max())
print('result', result[0].min())

depth_map = cv2.normalize(result[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
# resize to original image size
depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
# equalize histogram
# depth_map = (depth_map * 255).astype(np.uint8)
# depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

# Generate a mask to remove background using the depth map


# Resize the image to show it
image = cv2.resize(image, (image.shape[1]//5, image.shape[0]//5))
depth_map = cv2.resize(depth_map, (depth_map.shape[1]//5, depth_map.shape[0]//5))
# Convert to binary image
# depth_map = cv2.threshold(depth_map, 0.25, 1, cv2.THRESH_BINARY)[1]
cv2.imshow("image", image)
cv2.imshow("depth_map", depth_map)
cv2.waitKey(0)

# Generate mask to remove background
# mask = result[0] > 0.5
# Apply mask to image
# result[0][mask] = 0


# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].imshow(image)
# ax[1].imshow(result[0], cmap="magma")
# plt.show()