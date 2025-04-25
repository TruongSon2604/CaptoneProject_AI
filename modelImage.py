# import torch
# from torchvision import models, transforms

# from PIL import Image

# # Load ResNet50 bỏ lớp fully connected cuối
# model = models.resnet50(pretrained=True)
# model = torch.nn.Sequential(*list(model.children())[:-1])  # bỏ FC cuối
# model.eval()

# # Tiền xử lý ảnh
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# def extract_feature(img_path: str):
#     image = Image.open(img_path).convert("RGB")
#     input_tensor = preprocess(image).unsqueeze(0)  # (1, 3, 224, 224)

#     with torch.no_grad():
#         features = model(input_tensor).squeeze()  # (2048,)
#     return features / features.norm()  # normalize



import torch
from torchvision import models, transforms
from PIL import Image

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval() 

# Tiền xử lý ảnh
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# def extract_feature(img_path: str):
#     try:
#         image = Image.open(img_path).convert("RGB")
#     except Exception as e:
#         print(f"Error opening image: {e}")
#         return None

#     input_tensor = preprocess(image).unsqueeze(0) 

#     with torch.no_grad():
#         features = model(input_tensor).squeeze()  
    
#     return features / features.norm()

def extract_feature(img_path: str):
    try:
        # Load mô hình chỉ khi cần (lazy load)
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()

        # Tiền xử lý ảnh
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Mở ảnh
        image = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)  # (1, 3, 224, 224)

        with torch.no_grad():
            features = model(input_tensor).squeeze()  # (512,)

        return features / features.norm()  # normalize

    except Exception as e:
        print(f"Error in extract_feature: {e}")
        return None