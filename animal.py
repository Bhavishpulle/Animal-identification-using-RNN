import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
# Load the pre-trained ResNet50 model
model = ResNet50()
#Load and preprocess the image
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
print("size of the array : ",img_array.shape)
img_array = np.expand_dims(img_array, axis=0)
print("size of the array : ",img_array.shape)
img_array = resnet50.preprocess_input(img_array)
# Get predictions
predictions = model.predict(img_array)
# Decode predictions
decoded_predictions = resnet50.decode_predictions(predictions)
print(decoded_predictions)
# Display the top predictions
for i, (_, label, score) in enumerate(decoded_predictions[0]):
    print(i+1,":",label," ",score)
# Display the image with bounding boxes
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Draw bounding boxes on the image
for _, label, score in decoded_predictions[0]:
    if score > 0.8:  # Adjust the threshold based on your needs
        print("Object:",label,"Score:",score)
        cv2.putText(
            img,
            f"{label}: {score:.2f}",
            (10,20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            1,
            cv2.LINE_4,
        )
plt.imshow(img)
plt.show()
