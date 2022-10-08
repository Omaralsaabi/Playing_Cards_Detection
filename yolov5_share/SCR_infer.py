import cv2
import torch
from PIL import Image
import glob


# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp12/weights/best.pt')  # local model

# Image
imgs = []
for img in glob.glob("../test/images/*.jpg"):
    imgs.append(img)

print('first 10 images:')
print(imgs[:10])

print('inferring first 100 images: ')
# Inference
results = model(imgs[:100])

# results.pandas().xyxy[0]
results.save() 

# im1 = Image.open('zidane.jpg')  # PIL image
# im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)

# # Inference
# results = model([im1, im2], size=640) # batch of images

# # Results
# results.print()  
# results.save()  # or .show()

# results.xyxy[0]  # im1 predictions (tensor)
# results.pandas().xyxy[0]  # im1 predictions (pandas)