from os.path import join
import pickle
from pathlib import Path
import platform

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from models import model_2
from paths.paths import embeddings_and_labels_path, imgs_list_path


with open(embeddings_and_labels_path, 'rb') as in_file: # # arrays_1_1.npz for model_2_b-hard_50classes_31ep_020323 -> model: model_2_b-hard_ep25_50cls.pt
    npz_file = np.load(in_file)
    embeddings, labels = npz_file['embeddings'], npz_file['labels']

with open(imgs_list_path, 'rb') as in_file:
    imgs_path = pickle.load(in_file)

imgs_path_array = np.asarray(imgs_path)
if platform.system() == 'Linux':
    imgs_path_array = [img_path.as_posix() for img_path in imgs_path_array]


def calculate_embeddings(img_tensor: torch.tensor, model: torch.nn.Module):
    model.eval()
    with torch.no_grad():
        img_embedding = model(img_tensor)
        embeddings_norms = torch.norm(img_embedding, dim=1)
        img_embedding = torch.div(img_embedding, torch.unsqueeze(embeddings_norms, 1))
        
    return img_embedding


def calculate_distances(distance_function, query_embeddings: torch.tensor, ref_embeddings:torch.tensor, squared: bool = False):
    embeddings_concat = torch.concat([query_embeddings, ref_embeddings], dim=0)
    # distances = distance_function(embeddings_concat.numpy())[:, 1:]
    distances = distance_function(embeddings_concat, squared)[:, 1:]
    ordered_idx = np.argsort(distances, axis=1)
    distances_ordered = distances[ordered_idx]
    
    return distances_ordered, ordered_idx


def k_nearest_neighbors(idx_to_class_dict:dict, labels, ordered_idx, k: int = 3):
    ordered_labels = labels[ordered_idx]
    nearest_neighbors_classes = [idx_to_class_dict[label.item()] for label in ordered_labels[k + 1]]
    # nearest_neighbors_labels = np.array([label.item() for label in labels[row][1:n_neighbors + 1]])
    return nearest_neighbors_classes
        
   
