import os
import torch
from modelImage import extract_feature
from sklearn.metrics.pairwise import cosine_similarity

def load_features_from_folder(folder_path):
    features = []
    image_paths = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, filename)
            feature = extract_feature(path)
            features.append(feature)
            image_paths.append(filename)
    return torch.stack(features), image_paths

def find_similar_images(query_feature, all_features, image_paths, top_k=5):
    sims = cosine_similarity(query_feature.unsqueeze(0), all_features)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [(image_paths[i], float(sims[i])) for i in top_indices]
