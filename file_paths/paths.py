import os
import platform
from pathlib import Path

os_name = platform.system()
env_os_name = os.environ.get('OS_NAME', 'Not configured')

# idx_to_cls_dict_path = Path('./data/cls_to_idx_dict_50classes_v2.pkl')
# models_path = Path('./static/trained_models/')
# embeddings_and_labels_path = Path('./static/embeddings/sampled_train_arrays_model_2_b-hard-squared-dist_50classes_ep22_080323.npz')
# imgs_list_path = Path('./data/sampled_train_imgs_path_1_rel.pkl') 

if os_name == 'Linux' or env_os_name == 'Linux':
    idx_to_cls_dict_path = './data/cls_to_idx_dict_50classes_v2.pkl'
    models_path = './static/trained_models/'
    
    embeddings_and_labels_path = './static/embeddings/sampled_train_arrays_model_2_b-hard-squared-dist_50classes_ep22_080323.npz'
    imgs_list_path = './data/sampled_train_imgs_path_1_rel.pkl'
else:
    idx_to_cls_dict_path = '.\\data\\cls_to_idx_dict_50classes_v2.pkl'
    models_path = '.\\static\\trained_models\\'

    embeddings_and_labels_path = '.\\static\\embeddings\\sampled_train_arrays_model_2_b-hard-squared-dist_50classes_ep22_080323.npz'
    imgs_list_path = '.\\data\\sampled_train_imgs_path_1_rel.pkl'