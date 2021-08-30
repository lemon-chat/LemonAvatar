import os
import glob
import tqdm

from PIL import Image


if __name__ == "__main__":
    users = glob.glob("./data/*")
    print(f"用户数量：{len(users)}")
    artworks = glob.glob("./data/*/*.jpg")
    print(f"作品数量：{len(artworks)}")

    for img in tqdm.tqdm(artworks):
        try:
            image = Image.open(img)
        except: # PIL.UnidentifiedImageError
            print(img)
