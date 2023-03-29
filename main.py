from typing import Union
from PIL import Image

from fastapi import FastAPI
from fastapi import File, UploadFile
from fastapi.responses import FileResponse

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import euclidean_distances
from feature_extractors.models import preprocess, model_2, idx_to_class
from feature_extractors.embeddings_funcs import calculate_embeddings, calculate_distances, k_nearest_neighbors
from feature_extractors.embeddings_funcs import embeddings, labels, imgs_path_array
from schemas.preds_data_models import PredictionsKNN, TrainClasses, EmeddingInfo, ImagesIdxs
from utils.distances import euclidean_distance
from utils.utils import ImgDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embeddings = torch.tensor(embeddings)
# labels = torch.tensor(labels)

app = FastAPI(
    title='chilean spiders identifier',
    description='an API for k nearest neighbors for spiders images based on deep metric learning'
)

@app.get("/")
def read_root():
    train_classes = [v for k, v in idx_to_class.items()]
    return {
        'name': 'Chilean spiders identifiers',
        'train_set_size': 10_000,
        'n_classes': len(train_classes),
        'samples_per_classes': 300,
        'scope of classes': ['families', 'genera', 'species'],
        'feature_extractor': 'ResNet50',
        'embeddings_size': 1024
    }


@app.get("/get_train_classes/", response_model=TrainClasses)
def get_train_classes():
    train_classes = [v.replace('_', ' ') for k, v in idx_to_class.items()]
    return {"train_classes": train_classes, "n_train_classes": len(train_classes)}


@app.post("/send_img/", response_model=PredictionsKNN, tags=['predictions'])
async def create_upload_file(file: UploadFile):
    print(file.content_type)
    image = file.file
    image = torch.tensor(np.asarray(Image.open(image))).unsqueeze(dim=0).permute(0, 3, 1, 2)
    dataset = ImgDataset(image, transform=preprocess)
    dl = DataLoader(dataset, shuffle=False, batch_size=1)
    for img in dl:
        embedding = calculate_embeddings(img, model_2)
    
    _, ordered_idx = calculate_distances(
        euclidean_distance,
        embedding,
        ref_embeddings=embeddings,
        squared=True # no funciona para modelos entrenados con distancias al cuadrado
    )
    first_row_ordered_idx = ordered_idx[0]
    ordered_labels = labels[first_row_ordered_idx]
    first_k_nearest_neighbors = [idx_to_class[label].replace('_', ' ') for label in ordered_labels[:5]]
    
    return {'nearest_neighbors': first_k_nearest_neighbors, 'nearest_imgs_idx': first_row_ordered_idx[:5].tolist()}


# https://fastapi.tiangolo.com/advanced/custom-response/#fileresponse
# https://stackoverflow.com/questions/55873174/how-do-i-return-an-image-in-fastapi
@app.post("/send_nearest_imgs/", response_class=FileResponse, tags=['predictions'])
async def create_upload_file(idx: ImagesIdxs):
    idx = idx.dict()
    idx = np.asarray(idx['imgs_idxs'])
    selected_img_paths = imgs_path_array[idx]
    
    return selected_img_paths


@app.post("/get_img_embedding/", response_model=EmeddingInfo, tags=['predictions'])
async def create_upload_file(file: UploadFile):
    image = file.file
    image = torch.tensor(np.asarray(Image.open(image))).unsqueeze(dim=0).permute(0, 3, 1, 2)
    dataset = ImgDataset(image, transform=preprocess)
    dl = DataLoader(dataset, shuffle=False, batch_size=1)
    for img in dl:
        embedding = calculate_embeddings(img, model_2)
        
    return {'img_embedding': embedding.numpy()[0].tolist(), 'emb_shape': embedding.shape}