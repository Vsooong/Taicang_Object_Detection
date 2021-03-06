import os
import torchvision
import torch
import numpy as np
import sys
import matplotlib
import cv2
matplotlib.rc('figure', max_open_warning = 0)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T
import taicang_object_detection_train as train

COLORS1 = ['w', 'coral', 'lightgreen', 'red']
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


def get_prediction(img, model, threshold, device=torch.device('cpu')):
    trans = get_transform()
    image = trans(img).to(device)
    pred = model([image])
    pred_class = [int(i) for i in list(pred[0]['labels'].cpu().numpy())]

    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [index for index in range(len(pred_score)) if pred_score[index] > threshold[pred_class[index]]]
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


def object_detection_show(model, imgs, image_save_path, threshold):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
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


def object_detection_show_batch(model, imgs, image_save_path, batch_size=8, threshold=0.5):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    need_detect_images = sorted([pic for pic in imgs if not os.path.basename(pic).endswith('-det.jpg')])
    start_idx = 0
    length = len(need_detect_images)
    while start_idx < length:
        end_idx = min(length, start_idx + batch_size)
        batch_imgs = need_detect_images[start_idx:end_idx]
        open_images = []
        for impath in batch_imgs:
            image = Image.open(impath).convert("RGB")
            open_images.append(image)
        # predict result in one batch
        pboxes, pclass, pscore = get_prediction_batch(open_images, model, threshold, device)
        for index in range(len(batch_imgs)):
            boxes, pred_cls, pred_scr = pboxes[index], pclass[index], pscore[index]
            if len(boxes) > 0:
                name = os.path.basename(batch_imgs[index])
                pred = list(zip(pred_cls, np.around(pred_scr, decimals=3)))
                img = np.array(open_images[index])
                if Draw_Style == 1:
                    draw_bbox_matplot(img, boxes, pred_cls, pred_scr, image_save_path, name)

                else:  # Draw_Style == 2:
                    boxes.resize(len(boxes), 4)
                    boxes = boxes * img_scale_factor
                    out_image = draw_bbox(img, boxes, pred_cls, pred_scr)
                    im = Image.fromarray(out_image)
                    save_pth = os.path.join(image_save_path, name.replace('.jpg', '-det.jpg'))
                    im.save(save_pth)
                    im.close()

        start_idx += batch_size


def clear_test_samples(test_img_save_path):
    for i in sorted(os.listdir(test_img_save_path)):
        if i.endswith('det.jpg'):
            path = os.path.join(test_img_save_path, i)
            os.remove(path)



if __name__ == "__main__":
    INSTANCE_CATEGORY_NAMES = [
        '1', '2', '3', '4', '5'
    ]
    conf_thres = [0.2, 0.6, 0.6, 0.6]
    num_classes = train.num_classes
    assert len(conf_thres) + 1 == len(INSTANCE_CATEGORY_NAMES) == num_classes
    dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))

    img_size = train.img_size
    img_scale_factor = train.img_scale_factor
    Draw_Style = 1
    save_path = train.save_path
    # test_img_save_path = "/media/dl/HYX/samples/test/"\
    test_img_save_path = os.path.join(os.path.split(dirname)[0], 'test_image')
    # test_img_save_path='F:/data/Taicang/samples'
    clear_test_samples(test_img_save_path)
    test_img = [os.path.join(test_img_save_path, i) for i in os.listdir(test_img_save_path)]
    model = train.get_pretrained_model(save_path)
    object_detection_show(model, test_img, test_img_save_path, threshold=conf_thres)
