import os
import torchvision
import torch
import numpy as np
import tkinter
import matplotlib
import cv2

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
import detection.taicang_object_detection_train as train

COLORS1 = ['w', 'coral', 'lightgreen','red']
COLORS2 = np.asarray([[255., 255., 255.], [255., 128., 64.], [181., 230., 29.]])


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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


def get_prediction(img, model, threshold=0.5, device=torch.device('cpu')):
    trans = get_transform()
    image = trans(img).to(device)
    pred = model([image])
    pred_class = [int(i) for i in list(pred[0]['labels'].cpu().numpy())]

    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    pred_boxes = np.asarray(pred_boxes)
    pred_class = np.asarray(pred_class)
    pred_score = np.asarray(pred_score)
    pred_t = np.asarray(pred_t, dtype='int32')

    pred_boxes = pred_boxes[pred_t]
    pred_class = pred_class[pred_t]
    pred_score = pred_score[pred_t]
    return pred_boxes, pred_class, pred_score


def get_prediction_batch(imgs, model, threshold=0.5, device=torch.device('cpu')):
    open_images = []
    trans = get_transform()
    for i in imgs:
        open_images.append(trans(i))
    imgs = torch.stack(open_images, dim=0).to(device)
    pred = model(imgs)
    pboxes = []
    pclass = []
    pscore = []
    for i in range(imgs.size(0)):
        pred_class = [int(i) for i in list(pred[i]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in
                      list(pred[i]['boxes'].detach().cpu().numpy())]  # Bounding boxes
        pred_score = list(pred[i]['scores'].detach().cpu().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
        pred_boxes = np.asarray(pred_boxes)
        pred_class = np.asarray(pred_class)
        pred_score = np.asarray(pred_score)
        pred_t = np.asarray(pred_t, dtype='int32')
        pboxes.append(pred_boxes[pred_t])
        pclass.append(pred_class[pred_t])
        pscore.append(pred_score[pred_t])
    return pboxes, pclass, pscore


def draw_bbox(img, bbox, labels, confidence, colors=None, write_conf=True):
    global COLORS2
    bbox = np.asarray(bbox, dtype="int32")
    for i, label in enumerate(labels):
        if colors is None:
            color = COLORS2[int(label)]
        else:
            color = colors[int(label)]
        label = str(label)
        if write_conf:
            label += ': ' + str(format(confidence[i] * 100, '.0f')) + '%'

        cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color, 2)
        cv2.putText(img, label, (bbox[i][0], bbox[i][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def draw_bbox_matplot(img, bbox, labels, confidence, save_dir, name):
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12.8 * 1.32, 7.2 * 1.35), dpi=100)
    ax.imshow(img)
    for index, val in enumerate(bbox):
        (x1, y1), (x2, y2) = val
        x1, y1, x2, y2 = x1 * img_scale_factor, y1 * img_scale_factor, x2 * img_scale_factor, y2 * img_scale_factor
        box_h = y2 - y1
        box_w = x2 - x1
        color = COLORS1[int(labels[index])]
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(bbox)
        plt.text(x1, y1,
                 s=INSTANCE_CATEGORY_NAMES[int(labels[index])] + ': ' + str(100 * confidence[index])[:2] + '%',
                 color='white',
                 verticalalignment='top',
                 bbox={'color': color, 'pad': 0})
    plt.axis('off')
    save_pth = os.path.join(save_dir, name.replace('.jpg', '-det.jpg'))
    plt.savefig(save_pth, bbox_inches='tight', pad_inches=-0.25, transparent=True, dpi=100)
    print('Image saves to:', save_pth)
    plt.close(fig)


def object_detection_show(model_save_path, imgs, image_save_path, threshold=0.5):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()
    for pic in imgs:
        impath = pic
        name = os.path.basename(impath)
        if name.endswith('-det.jpg'):
            continue
        img = Image.open(impath).convert("RGB")
        # print('Image in: ', impath)
        boxes, pred_cls, pred_scr = get_prediction(img, model, threshold, device)  # Get predictions
        pred = list(zip(pred_cls, np.around(pred_scr, decimals=3)))
        # print(sorted(pred, key=lambda i: i[1], reverse=True))

        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        img = np.array(img)

        if len(boxes) > 0:
            if Draw_Style == 1:
                draw_bbox_matplot(img, boxes, pred_cls, pred_scr, image_save_path, name)

            else:  # Draw_Style == 2:
                boxes.resize(len(boxes), 4)
                boxes = boxes * img_scale_factor
                out_image = draw_bbox(img, boxes, pred_cls, pred_scr)
                im = Image.fromarray(out_image)
                save_pth = os.path.join(image_save_path, name.replace('.jpg', '-det.jpg'))
                im.save(save_pth)
                print('Image saves to:', save_pth)
                im.close()

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

if __name__ == "__main__":
    INSTANCE_CATEGORY_NAMES = [
        '1', '2', '3', '4', '5'
    ]
    num_classes = train.num_classes
    img_size = train.img_size
    img_scale_factor = train.img_scale_factor
    conf_thres = 0.3
    Draw_Style = 1
    save_path = train.save_path
    # test_img_save_path = "/media/dl/HYX/samples/test/"
    test_img_save_path='F:/data/Taicang/samples'
    test_img = [os.path.join(test_img_save_path, i) for i in os.listdir(test_img_save_path)]
    object_detection_show(save_path, test_img, test_img_save_path, threshold=conf_thres)
