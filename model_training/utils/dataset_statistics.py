import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from iosdata.water_segmentation import Water


'''Mean and standard deviation for the water_segmentation dataset'''
TAMP_OPEN_DOCK = ([0.48955124616622925, 0.47795844078063965, 0.49367910623550415], [0.24107402563095093, 0.2379443198442459, 0.24587485194206238])
MISC_TRAINING = ([0.4454203248023987, 0.4749860167503357, 0.4680652916431427], [0.2575828433036804, 0.2523757517337799, 0.2858140468597412])
TAMP_OPEN_DOCK_MISC_TRN = ([0.4522751271724701, 0.4754515588283539, 0.4720318019390106], [0.2556144595146179, 0.2502172291278839, 0.280158668756485])



def get_mean_std(loader):
    '''
    Function to calculate standard deviation and mean of a given dataset.
    Needed to normalize the data to 0 mean and standard deviation to 1.

    Taken from Aladdin Persson
    https://www.youtube.com/watch?v=y6IEcEBRZks
    '''
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0,0,0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std= (channels_squared_sum/num_batches - mean**2)**0.5

    return mean.tolist(), std.tolist()



# Usage:
'''  
# Can be used for verification of 0 mean and 1 std
data_transforms = {
    'train': transforms.Compose([
        transforms.Normalize([0.4454, 0.4750, 0.4681],[0.2576, 0.2524, 0.2858])
    ]),
    'val': transforms.Compose([
        transforms.Normalize([0.4454, 0.4750, 0.4681],[0.2576, 0.2524, 0.2858])
    ]),
}

dataset = Water('/home/konstantin/projects/WaterDataset', img_size=(960,640))
data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=False, num_workers=4)

print(get_mean_std(data_loader))
print(len(dataset))
'''
