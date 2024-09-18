from lavis.models import load_model_and_preprocess
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tt
import torch
import numpy as np

import warnings
warnings.filterwarnings('ignore')

model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
cifar_train = torchvision.datasets.CIFAR100(root='/content/drive/MyDrive/GMM_DW/datasets/CIFAR100/', train=True, download=True)

X = np.concatenate([np.asarray(cifar_train[i][0]) for i in range(len(cifar_train))])

mean = np.mean(X, axis = (0, 1)) / 255
std = np.std(X, axis = (0, 1)) / 255

mean = mean.tolist()
std = std.tolist()

transform_train = tt.Compose([tt.RandomCrop(32
                                            , padding = 4
                                            #, padding_mode = 'reflect'
                                            ),
                              tt.RandomHorizontalFlip(),
                              tt.ToTensor(),
                              ])
transform_test = tt.Compose([tt.ToTensor(),
                             ])


training_set = torchvision.datasets.CIFAR100('/content/drive/MyDrive/GMM_DW/datasets/CIFAR100/train', train = True, download = True, transform = transform_train)
test_set = torchvision.datasets.CIFAR100('/content/drive/MyDrive/GMM_DW/datasets/CIFAR100/test', train = False, download = True, transform = transform_test)

trainloader = torch.utils.data.DataLoader(training_set, batch_size = 128, shuffle = True, num_workers = 2)
testloader = torch.utils.data.DataLoader(test_set, batch_size = 100, shuffle = False, num_workers = 2)


class_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 
                'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 
                'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 
                'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 
                'wolf', 'woman', 'worm']
class_dict = {}

for i in range(len(class_labels)):
  class_dict[i] = ''




for batch in trainloader:

  for image_index in range(len(batch)):

    image_transpose = (batch[0][image_index].numpy().transpose(1, 2, 0))
    label_index = batch[1][image_index].item()

    im = Image.fromarray((image_transpose * 255).astype(np.uint8))
    image = vis_processors['eval'](im).unsqueeze(0).to(device)

    im = im.resize((128, 128), Image.Resampling.LANCZOS)
    #display(im)
    

    generated_caption = model.generate({'image' : image, 'prompt' : 'Question: What is this an image of? Answer: '})

    class_dict[label_index] += (generated_caption[0] + '. ')

    print('Image Label: ', class_labels[label_index])
    print('Generated Caption:', generated_caption)
    print()

    break


## Display class labels and their generated captions
for key in class_dict:
  print(class_labels[key] + ' : ' + class_dict[key])
