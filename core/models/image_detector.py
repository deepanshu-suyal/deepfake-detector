import torch
import timm
from torchvision import transforms
from datetime import datetime


class ImageDetector:

    def __init__(self):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=2
        )

        self.model.eval()

        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image):

        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():

            outputs = self.model(tensor)

            probs = torch.softmax(outputs, dim=1)

        real_prob = probs[0][0].item()

        ai_prob = probs[0][1].item()

        return {
            "is_ai_generated": ai_prob > real_prob,
            "ai_probability": ai_prob,
            "real_probability": real_prob,
            "confidence": max(ai_prob, real_prob),
            "timestamp": datetime.now().isoformat()
        }