import os
import glob
if __name__ == "__main__":
    users = glob.glob("./data/*")
    print(f"用户数量：{len(users)}")
    artworks = glob.glob("./data/*/*.jpg")
    print(f"作品数量：{len(artworks)}")
