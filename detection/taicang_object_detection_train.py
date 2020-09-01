import os
import torchvision
import torch
import xml.dom.minidom
import tkinter
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from detection.engine import train_one_epoch, evaluate
import detection.utils as utils
import torchvision.transforms as T
import random
from torch.utils.data import Dataset, DataLoader


# from torchsummary import summary


class TaiCangDataset(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(img_root_dir)))
        self.labels = list(sorted(os.listdir(ans_root_dir)))
        self.img_scale_factor = img_scale_factor

    def __getitem__(self, idx):
        # print("image _idx", idx)
        img_path = os.path.join(img_root_dir, self.imgs[idx])
        label_path = os.path.join(ans_root_dir, self.labels[idx])
        img = Image.open(img_path).convert("RGB")
        dom = xml.dom.minidom.parse(label_path)
        root = dom.documentElement
        height = int(root.getElementsByTagName('height')[0].firstChild.data)  # Image height
        width = int(root.getElementsByTagName('width')[0].firstChild.data)  # Image width

        boxes = []
        labels = []

        for index in range(len(root.getElementsByTagName('object'))):
            xmin = float(root.getElementsByTagName('xmin')[
                             index].firstChild.data)  # List of normalized left x coordinates in bounding box (1 per box)
            ymin = float(root.getElementsByTagName('ymin')[
                             index].firstChild.data)  # List of normalized left x coordinates in bounding box (1 per box)
            xmax = float(root.getElementsByTagName('xmax')[
                             index].firstChild.data)  # List of normalized left x coordinates in bounding box (1 per box)
            ymax = float(root.getElementsByTagName('ymax')[
                             index].firstChild.data)  # List of normalized left x coordinates in bounding box (1 per box)
            cls = int(root.getElementsByTagName('name')[index].firstChild.data) - 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls)

        boxes = torch.as_tensor(boxes, dtype=torch.float32) / self.img_scale_factor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        image_name = torch.tensor(int(os.path.basename(img_path).split('.')[0]))
        label_name = torch.tensor(int(os.path.basename(label_path).split('.')[0]))
        num_box = labels.size(0)
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # masks = torch.zeros(size=(num_box, height, width), dtype=torch.uint8)
            iscrowd = torch.zeros(size=(num_box,), dtype=torch.int64)
        except:
            print(label_path)
            print(boxes)
            print(labels)
        else:
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            target['image_path'] = image_name
            # target["masks"] = masks
            if self.transforms is not None:
                img = self.transforms(img)
            return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes, pretrained=True):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train=True):
    transforms = []
    transforms.append(T.Resize(img_size))
    transforms.append(T.ToTensor())
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train_model(model):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device: ', device)

    dataset = TaiCangDataset(get_transform(train=True))
    dataset_test = TaiCangDataset(get_transform(train=False))

    # split the dataset in train and test set

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices)
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])
    # dataset_test = torch.utils.data.Subset(dataset_test, indices[:100])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False,
        collate_fn=utils.collate_fn)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=0.001, amsgrad=False)
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=log_step)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        # if (epoch + 1) % 5 == 0:
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), save_path)


def get_transfer_model():
    # get the model using our helper function
    model = get_model_instance_segmentation(3, False)
    model.load_state_dict(torch.load('../Save/model-4278.pth'))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_pretrained_model(pretrain_path):
    model = get_model_instance_segmentation(num_classes, False)
    model.load_state_dict(torch.load(pretrain_path))
    return model


# img_root_dir = '/home/dl/data/taicang/data10000/img'
# ans_root_dir = '/home/dl/data/taicang/data10000/label'
img_root_dir = "/media/dl/HYX/samples/img"
ans_root_dir = "/media/dl/HYX/samples/label"
pretrained_path = '../Save/model_0901.pth'
# save_path = '/home/dl/data/taicang/data10000/OD_pytorch/Save/ssd.pth'
save_path = '../Save/model_0901-2.pth'
img_size = (360, 640)
img_scale_factor = 2
num_classes = 5

if __name__ == "__main__":
    test_size = 100
    num_epochs = 10
    log_step = 15
    batch_size = 4
    model = get_pretrained_model(pretrained_path)
    train_model(model)

    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # summary(model,(3,360,640))
