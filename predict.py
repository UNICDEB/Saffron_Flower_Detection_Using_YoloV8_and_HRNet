import cv2
import matplotlib.pyplot as plt
from train_hrnet_pluck import predict_on_crop

# Paths
ckpt = "hrnet_pluck_best.pth"
img_path = "E:/Project_Work/2025/Saffron_Project/Github_Code/Saffron_Detection/YoloV8+HRNet/YoloV8_Result_Object_detection/test/images_field/image_36.jpeg"

# Run prediction
px, py = predict_on_crop(ckpt, img_path, img_size=256, device="cpu")
print("Normalized pluck:", px, py)

# --- Display the image with predicted point ---
# Load the image
img = cv2.imread(img_path)
h, w = img.shape[:2]

# Convert normalized coords -> pixel coords
x_abs = int(px * w)
y_abs = int(py * h)

# Draw circle
cv2.circle(img, (x_abs, y_abs), 5, (0, 0, 255), -1)

# Show using matplotlib (RGB)
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Predicted Pluck Point ({x_abs}, {y_abs})")
plt.axis("off")
plt.show()
