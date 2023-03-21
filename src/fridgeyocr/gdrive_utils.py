import requests
import gdown
import os

def download_model_weight_from_gdrive(id, destination):
    URL = f"https://drive.google.com/uc?id={id}"
    gdown.download(URL, destination, quiet=False, fuzzy=True)
    print("DOWNLOADED MODEL WEIGHT")
    
# https://drive.google.com/file/d/1XF76z5iRhYxrbAiLIddJC5X5dJkmumDh/view?usp=share_link
# https://drive.google.com/file/d/1XpujSr2yV35E-25Gs8FmxOJDgYthehg3/view?usp=share_link
if __name__ == "__main__":
    gdrive_test= '/home/guest/speaking_fridgey/gdrive_test'
    destination = os.path.join(gdrive_test, 'ctpn.pt')
    download_model_weight_from_gdrive(id='1XF76z5iRhYxrbAiLIddJC5X5dJkmumDh', \
                                      destination=destination)