from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, txt_path, num_class, transforms=None):
        super(MyDataset, self).__init__()
        images = []
        labels = []
        with open(txt_path, 'r') as f:
            for line in f:
                if int(line.split('\\')[1]) >= num_class:
                    break
                line = line.strip('\n') # remove newline char
                images.append(line) # append image path from txt file
                labels.append(int(line.split('\\')[1])) # the directory name is ground truth
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB') # convert image to RGB to fit initial channel size
        label = self.labels[index]
        if self.transforms is not None:
            image = self.transforms(image) # perform data augmentation on an image
        return image, label

    def __len__(self):
        return len(self.labels)