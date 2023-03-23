from os.path import join
import pickle

import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from file_paths.paths import idx_to_cls_dict_path, models_path
from utils.utils import download_from_drive

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_states(save_state_path:str, model:torch.nn.Module):
    # map_location=torch.device('cpu') only in case of a cpu
    checkpoint = torch.load(save_state_path, map_location=torch.device('cpu')) 
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, epoch, loss

with open(idx_to_cls_dict_path, 'rb') as in_f:
    idx_to_class = pickle.load(in_f)

weights = ResNet50_Weights.DEFAULT # Cambiar el modelo pre-entrenado según sea necesario
preprocess = weights.transforms()
preprocess.to(device)

model_2 = resnet50()
num_ftrs = model_2.fc.in_features # dimensión de entrada de la última capa "fc"
linear_layers = nn.Sequential()
linear_layers.add_module('linear1', nn.Linear(num_ftrs, num_ftrs))
linear_layers.add_module('lrelu_l1', nn.LeakyReLU())
linear_layers.add_module('dropout1', nn.Dropout(p=0.2))
linear_layers.add_module('linear2', nn.Linear(num_ftrs, 1024))
linear_layers.add_module('lrelu_l2', nn.LeakyReLU())
linear_layers.add_module('dropout1', nn.Dropout(p=0.2))
linear_layers.add_module('linear3', nn.Linear(1024, 1024))
linear_layers.add_module('lrelu_l3', nn.LeakyReLU())
linear_layers.add_module('linear4', nn.Linear(1024, 1024))
model_2.fc = linear_layers
model_2.to(device)

selected_file_name = 'model_2_b-hard-squared-dist_ep22_50cls.pt' # cambiar el nombre del modelo según sea necesario
download_from_drive("https://drive.google.com/uc?id=11ZnGZ_3yNjmvg9pGScr7oVOewzo6vtJ4", file_path=selected_file_name)
# models_path = r'./static/trained_models/'
model_2, _, _ = load_states(selected_file_name, model_2) # join(models_path, selected_file_name)
for param in model_2.parameters():
    param.requires_grad = False
    
# models used

# dir: model_2_b-hard_50classes_31ep_020323 -> model: model_2_b-hard_ep25_50cls.pt
# Se debe tener en cuenta: embeddings producidos por el modelo

# dir: model_2_b-hard-squared-dist_50classes_29ep_080323 -> model: model_2_b-hard-squared-dist_ep22_50cls.pt
# Se debe tener en cuenta: argumento square y embeddings producidos por el modelo