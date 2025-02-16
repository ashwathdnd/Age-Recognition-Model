import torch
from model import AgeNet
from PIL import Image
from torchvision import transforms

def predict_age(image_path):
    # Load the model
    model = AgeNet()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Setup image transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(image)
    
    return prediction.item()

# Test it on an image
if __name__ == "__main__":
    image_path = "baby.jpg"  # Replace with your test image
    predicted_age = predict_age(image_path)
    print(f"Predicted Age: {predicted_age:.1f} years")