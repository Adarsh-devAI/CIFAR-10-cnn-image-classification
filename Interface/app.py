import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import torch.nn.functional as F
import numpy as np

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class names
classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# model definition (SAME as training)
class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepCNN, self).__init__()

        def conv_block(in_c, out_c, pool=False):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2,2))
            return nn.Sequential(*layers)

        self.layer1 = conv_block(3, 64, pool=True)
        self.layer2 = conv_block(64, 128, pool=True)
        self.layer3 = conv_block(128, 256, pool=False)
        self.layer4 = conv_block(256, 256, pool=True)
        self.layer5 = conv_block(256, 512, pool=False)
        self.layer6 = conv_block(512, 512, pool=True)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.classifier(x)
        return x

# load model
model = DeepCNN().to(device)
model.load_state_dict(torch.load("../model/best_deepcnn.pth", map_location=device))
model.eval()

# transform
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

test_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def predict(img):
    img = test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)[0]
    return {classes[i]: float(probs[i]) for i in range(10)}

# gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil'),
    outputs=gr.Label(num_top_classes=5),
    title="Deep CNN CIFAR-10 Classifier",
    description="Upload an image and CNN will classify it"
)

interface.launch()
iface.launch(share=True)