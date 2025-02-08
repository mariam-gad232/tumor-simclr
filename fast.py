from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io
import torchvision.transforms as transforms

app = FastAPI()

# تحميل نموذج PyTorch
model = torch.load("simclr_brain_tumor.pt", map_location=torch.device('cpu'))
model.eval()

# تجهيز الصورة للتنبؤ
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    class_labels = ["No Tumor", "Tumor Detected"]
    return {"prediction": class_labels[predicted_class]}

# لتشغيل السيرفر محليًا
# uvicorn main:app --host 0.0.0.0 --port 8000
