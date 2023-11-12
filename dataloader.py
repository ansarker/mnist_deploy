from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MnistFashionDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        self.dataroot = './data/fashion-mnist'
        self.data = pd.read_csv(os.path.join(self.dataroot, csv_file))
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data.iloc[index, 1:].values.astype('float32').reshape(28, 28)
        image = self.transform(image)
        label = self.data.iloc[index, 0]

        return image, label

def show_batch(images, labels, classes):
    num_images = images.shape[0]
    num_cols = 8
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig = plt.figure(figsize = (16, 16))
    for i in range(num_images):
        image_np = images[i].numpy()
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(np.transpose(image_np, (1, 2, 0)), cmap='gray')
        # plt.imsave(f'./crud_images/fashion_img_{i}.png', np.transpose(image_np, (1, 2, 0)).squeeze(2), cmap='gray')
        plt.title("{}".format(classes[labels[i]]), fontsize=14)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./runs/image_visualization.png')

def main():
    train_data = MnistFashionDataset('fashion-mnist_train.csv')
    test_data = MnistFashionDataset('fashion-mnist_test.csv')

    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    train_data = iter(train_loader)
    images, labels = next(train_data)

    show_batch(images, labels, classes)
    

if __name__ == "__main__":
    main()

def main_ana():
    train_data = pd.read_csv('./data/fashion-mnist/fashion-mnist_train.csv').to_numpy()
    labels = train_data[:, 0]
    pixels = train_data[:, 1:]
    images = pixels.reshape(-1, 28, 28)

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    num_images = 10
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(classes[int(labels[i])])
        axes[i].axis('off')

    plt.savefig('./runs/image_visualization.png')
