from pydantic import BaseModel

class TrainClasses(BaseModel):
    train_classes: list[str]
    n_train_classes: int

class PredictionsKNN(BaseModel):
    nearest_neighbors: list[str]
    nearest_imgs_idx: list[int]
    
class EmeddingInfo(BaseModel):
    img_embedding: list[float]
    emb_shape: list[int]
    
class ImagesIdxs(BaseModel):
    imgs_idxs: int