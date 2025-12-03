import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from pathlib import Path
import io

class ModelWrapper:
    """Wrapper class to standardize model interface"""
    def __init__(self, model, transform, device='cpu'):
        self.model = model
        self.transform = transform
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def predict(self, input_data):
        """
        Predict from image bytes or PIL Image
        Returns: numpy array of shape (batch_size, num_classes) with probabilities
        """
        with torch.no_grad():
            if isinstance(input_data, bytes):
                img = Image.open(io.BytesIO(input_data)).convert("RGB")
            elif isinstance(input_data, Image.Image):
                img = input_data
            elif isinstance(input_data, np.ndarray):
                if len(input_data.shape) == 4:
                    input_data = input_data[0]
                if input_data.max() <= 1.0:
                    input_data = (input_data * 255).astype(np.uint8)
                img = Image.fromarray(input_data)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            return probs.cpu().numpy()

def load_xception_model():
    """Load custom Xception model"""
    from models.custom_xception import MiniXception
    
    model_path = Path(__file__).resolve().parent / "models" / "best_xception_overall.pth"
    model = MiniXception(num_classes=3)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu', weights_only=True))
    
    transform = transforms.Compose([
        transforms.Resize((1024, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    
    return ModelWrapper(model, transform)

def load_resnet50_model():
    """Load ResNet50 model"""
    from torchvision.models import resnet50
    import torch.nn as nn
    
    model_path = Path(__file__).resolve().parent / "models" / "best_resnet50_overall.pth"
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu', weights_only=True))
    
    transform = transforms.Compose([
        transforms.Resize((1024, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    return ModelWrapper(model, transform)

def load_efficientnet_model():
    """Load EfficientNetB0 model"""
    from torchvision.models import efficientnet_b0
    import torch.nn as nn
    
    model_path = Path(__file__).resolve().parent / "models" / "best_efficientnetb0_overall.pth"
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu', weights_only=True))
    
    transform = transforms.Compose([
        transforms.Resize((1024, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    return ModelWrapper(model, transform)

def load_mobilenet_model():
    """Load MobileNetV3 model"""
    from torchvision.models import mobilenet_v3_large
    import torch.nn as nn
    
    model_path = Path(__file__).resolve().parent / "models" / "best_mobilenetv3_overall.pth"
    model = mobilenet_v3_large(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 3)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu', weights_only=True))
    
    transform = transforms.Compose([
        transforms.Resize((1024, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    return ModelWrapper(model, transform)

def get_all_models():
    """Load all available models"""
    models = {}
    try:
        models['xception'] = load_xception_model()
        print("✓ Loaded Xception model")
    except Exception as e:
        print(f"✗ Failed to load Xception: {e}")
    
    try:
        models['resnet50'] = load_resnet50_model()
        print("✓ Loaded ResNet50 model")
    except Exception as e:
        print(f"✗ Failed to load ResNet50: {e}")
    
    try:
        models['efficientnet'] = load_efficientnet_model()
        print("✓ Loaded EfficientNetB0 model")
    except Exception as e:
        print(f"✗ Failed to load EfficientNetB0: {e}")
    
    try:
        models['mobilenet'] = load_mobilenet_model()
        print("✓ Loaded MobileNetV3 model")
    except Exception as e:
        print(f"✗ Failed to load MobileNetV3: {e}")
    
    return models