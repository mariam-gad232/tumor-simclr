from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size of the model
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load an image from the internet or local file
image_path = "path_to_your_image.jpg"  # Replace with the path to your image
image = Image.open(image_path).convert('RGB')  # Open and convert to RGB

# Preprocess the image
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension

# Load the trained model
model = SimCLR()
model.load_state_dict(torch.load('simclr_brain_tumor.pth'))
model.eval()  # Set the model to evaluation mode

# Move the input to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_batch = input_batch.to(device)
model = model.to(device)

# Extract features
with torch.no_grad():
    features = model.extract_features(input_batch)

# Use the classifier to get predictions
classifier = nn.Linear(512, 4).to(device)  # Replace 512 with the feature size and 4 with the number of classes
classifier.load_state_dict(model.classifier.state_dict())  # Load the trained classifier weights
outputs = classifier(features)

# Get the predicted class
_, predicted_class = torch.max(outputs, 1)
predicted_class = predicted_class.item()

# Define class names (replace with your actual class names)
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Get the predicted class name
predicted_class_name = class_names[predicted_class]
print(f"Predicted Class: {predicted_class_name}")

# Display the image
plt.imshow(image)
plt.title(f"Predicted: {predicted_class_name}")
plt.axis('off')
plt.show()