import sys

import gdown

def download_from_drive(url:str, file_path:str):
    gdown.download(url, file_path, quiet=False)
    print('\nFile downloaded')
    print('**'*20)
    
url = sys.argv[1]
states_file_name = 'model_2_b-hard-squared-dist_ep22_50cls.pt'
download_from_drive(url=url, file_path=states_file_name)