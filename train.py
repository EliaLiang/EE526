
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":

    # Cuda = True
    Cuda = False

    classes_path   = './98_classes .txt'
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model_path      = 'model_data/ep095-loss1.521-val_loss0.839.pth'

    input_shape     = [416, 416]
    backbone        = "mobilenetv3"
    pretrained      = True

    mosaic              = False
    Cosine_lr           = False
    label_smoothing     = 0

    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 16
    Freeze_lr           = 1e-3

    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 8
    Unfreeze_lr         = 1e-4
    Freeze_Train        = True

    num_workers         = 0

    train_annotation_path   = '2007_train.txt'    #get image path and label
    val_annotation_path     = '2007_val.txt'
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)


    model = YoloBody(anchors_mask, num_classes, backbone = backbone, pretrained = pretrained)#create yolo model
    if not pretrained:
        weights_init(model)
    if model_path != '':

        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    loss_history = LossHistory("logs/")

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)


    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch
        print("number of training set：", num_train)
        print("number of validation set：", num_val)
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small for training, please expand the dataset.")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train = True)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)


        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        min_loss_train = 100000.0
        min_loss = 100000.0
        best_model = None
        for epoch in range(start_epoch, end_epoch):
            train_loss, val_loss, current_model = fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            if val_loss < min_loss:
                min_loss = val_loss
                min_loss_train = train_loss
                best_model = current_model
            lr_scheduler.step()

        torch.save(best_model, 'logs/loss%.3f-val_loss%.3f.pth' % (min_loss_train / epoch_step, min_loss / epoch_step_val))
            
    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small for training, please expand the dataset.")
        
        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, mosaic=mosaic, train = True)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, mosaic=False, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)

        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = True

        min_loss_train = 100000.0
        min_loss = 100000.0
        best_model = None
        for epoch in range(start_epoch, end_epoch):
            train_loss, val_loss, current_model = fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch,
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
            if val_loss < min_loss:
                min_loss = val_loss
                min_loss_train = train_loss
                best_model = current_model
            lr_scheduler.step()

        torch.save(best_model,'logs/loss%.3f-val_loss%.3f.pth' % (min_loss_train / epoch_step, min_loss / epoch_step_val))
