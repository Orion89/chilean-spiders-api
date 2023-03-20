import platform
from pathlib import Path

os_name = platform.system()

idx_to_cls_dict_path = Path('./data/cls_to_idx_dict_50classes_v2.pkl')
models_path = Path('./static/trained_models/')

embeddings_and_labels_path = Path('./static/embeddings/sampled_train_arrays_model_2_b-hard-squared-dist_50classes_ep22_080323.npz')
imgs_list_path = Path('./data/sampled_train_imgs_path_1_rel.pkl') 

if os_name == 'Linux':
    idx_to_cls_dict_path = idx_to_cls_dict_path.as_posix()
    models_path = models_path.as_posix()
    
    embeddings_and_labels_path = embeddings_and_labels_path.as_posix()
    imgs_list_path = imgs_list_path.as_posix()