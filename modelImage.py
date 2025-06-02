import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval() 

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_feature(img_path: str):
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    input_tensor = preprocess(image).unsqueeze(0) 

    with torch.no_grad():
        features = model(input_tensor).squeeze()  
    
    return features / features.norm()

# def extract_feature(img_path: str):
#     try:
#         model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#         model = torch.nn.Sequential(*list(model.children())[:-1])
#         model.eval()
#         preprocess = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])
#         image = Image.open(img_path).convert("RGB")
#         input_tensor = preprocess(image).unsqueeze(0)

#         with torch.no_grad():
#             features = model(input_tensor).squeeze() 

#         return features / features.norm()

#     except Exception as e:
#         print(f"Error in extract_feature: {e}")
#         return None