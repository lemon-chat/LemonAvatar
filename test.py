from dataset import PixivDataset

dataset = PixivDataset("./data")

img = dataset[0]
print(img)