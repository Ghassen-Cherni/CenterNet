import pandas as pd
import torch.optim as optimizer
import torch
from tqdm import tqdm
import gc
import os
from args_parse import *
from preprocess import *
import matplotlib.pyplot as plt
from model import *
from dataloaders import *
from loss_function import *
from leslie_experience import *
import pickle
from sklearn.utils import shuffle
from average_precision import *
import numpy as np
args = parse_args()


def train_model(epoch, optimizer, scheduler=None, epochs=10, logs=[]):
    model.train()
    print('epochs {}/{} '.format(epoch + 1, epochs))
    # print('learning rate {} '.format(optimizer.state_dict()['param_groups'][0]['lr']))
    running_loss = 0.0
    running_mask = 0.0
    running_regr = 0.0
    running_offset = 0.0
    running_loss_val = 0.0
    t = tqdm(train_loader)
    s = tqdm(validation_loader)
    rd = np.random.rand()
    precision_train = 0.0
    precision_val = 0.0
    for idx, (img, index, hm, regr, offset) in enumerate(t):
        # send to gpu
        img = img.to(device)
        hm_gt = hm.to(device)
        regr_gt = regr.to(device)
        offset_gt = offset.to(device)
        # set opt
        ## set_to_none=True apparently optimizes the memory usage
        optimizer.zero_grad(set_to_none=True)

        # run model
        hm, regr, offset = model(img)
        preds = torch.cat((hm, regr, offset), 1)

        loss, mask_loss, regr_loss, offset_loss = centerloss(preds, hm_gt, regr_gt, offset_gt)
        # misc
        running_loss += loss
        running_mask += mask_loss
        running_regr += regr_loss
        running_offset += offset_loss

        ### Precision calculation
        reg = torch.cat((offset_gt, regr_gt), 1)
        reg_pred = torch.cat([offset, regr], 1)
        precision_train += get_average_precision(hm_gt.detach(), reg.detach(), hm.detach(), reg_pred.detach(), 5,
                                                 0.4) * len(index) / len(train_loader)

        loss.backward()
        optimizer.step()

        t.set_description(
            f't (l={running_loss / (idx + 1):.3f})(m={running_mask / (idx + 1):.4f})(r={running_regr / (idx + 1):.4f})(o={running_offset / (idx + 1):.4f})')

    print('learning rate : ', optimizer.param_groups[0]['lr'])
    print('train loss : {:.4f}'.format(running_loss / len(train_loader)))
    print('maskloss : {:.4f}'.format(running_mask / (len(train_loader))))
    print('regrloss : {:.4f}'.format(running_regr / (len(train_loader))))
    print('offsetloss : {:.4f}'.format(running_offset / (len(train_loader))))
    print('precision : {:.4f}'.format(precision_train))
    # save logs

    with torch.no_grad():
        for idx, (img, index, hm, regr, offset) in enumerate(s):
            # send to gpu
            img = img.to(device)
            hm_gt = hm.to(device)
            regr_gt = regr.to(device)
            offset_gt = offset.to(device)

            # run model
            hm, regr, offset = model(img)
            preds = torch.cat((hm, regr, offset), 1)

            loss_validation, mask_loss_validation, regr_loss_validation, offset_loss_validation = centerloss(preds,
                                                                                                             hm_gt,
                                                                                                             regr_gt,
                                                                                                             offset_gt)
            running_loss_val += loss_validation
            reg = torch.cat((offset_gt, regr_gt), 1)
            reg_pred = torch.cat([offset, regr], 1)
            precision_val += get_average_precision(hm_gt, reg, hm, reg_pred, 5, 0.4) * len(index) / len(
                validation_loader)

    log_epoch = {'epoch': epoch + 1, 'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                 'loss': running_loss / len(train_loader), "mask": running_mask / (len(train_loader)),
                 "regr": running_regr / (len(train_loader)), 'offset': running_offset / (len(train_loader)),
                 'precision_train': precision_train
                ,'loss_validation': running_loss_val / (len(validation_loader)), 'precision_val': precision_val}

    logs.append(log_epoch)
    losts.append([epoch + 1
                 , optimizer.state_dict()['param_groups'][0]['lr']
                 , running_loss / len(train_loader)
                 , running_mask / (len(train_loader))
                 , running_regr / (len(train_loader))
                 , running_offset / (len(train_loader))
                 , precision_train
                 , running_loss_val / (len(validation_loader))
                 , precision_val]
                 )
    if scheduler is not None:
        scheduler.step()
    return logs


def train_loop(model, scheduler_name=None, epochs=10):
    if scheduler_name is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = None
    else:
        optimizer = optim.SGD(model.parameters(), lr=1e-2)
        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, eta_min=1e-3)
        elif scheduler_name == 'one_cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, epochs=epochs, steps_per_epoch=181,
                                                            three_phase=True)
    logs = []
    for epoch in range(epochs):
        logs = train_model(epoch, scheduler=scheduler, epochs=epochs, optimizer=optimizer, logs=logs)
        losts_arr = np.array(losts)
        fig, axs = plt.subplots(3, 3)
        axs[0, 0].plot(losts_arr[:,0], losts_arr[:,1])
        axs[0, 0].set_title('lr')
        axs[0, 1].plot(losts_arr[:,0], losts_arr[:,2])
        axs[0, 1].set_title('loss')
        axs[0, 2].plot(losts_arr[:,0], losts_arr[:,3])
        axs[0, 2].set_title('mask')
        axs[1, 0].plot(losts_arr[:,0], losts_arr[:,4])
        axs[1, 0].set_title('rg')
        axs[1, 1].plot(losts_arr[:,0], losts_arr[:,5])
        axs[1, 1].set_title('offset')
        axs[1, 2].plot(losts_arr[:,0], losts_arr[:,6])
        axs[1, 2].set_title('precision_train')
        axs[2, 0].plot(losts_arr[:,0], losts_arr[:,7])
        axs[2, 0].set_title('loss_validation')
        axs[2, 1].plot(losts_arr[:,0], losts_arr[:,8])
        axs[2, 1].set_title('precision_val')
        fig.savefig("Losts.png")

        torch.save(model.state_dict(), "model")
        gc.collect()
    return logs
# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    print(f"args.epoch = {args.epoch}, args.lr={args.lr}, args.batch_size={args.batch_size}, args.num_workers={args.num_workers}\n"
          f"args.train_file={args.train_file}, args.train_dir={args.train_dir}, args.test_dir={args.test_dir}\n"
          f"args.augment_data={args.augment_data}, args.leslie={args.leslie}, args.lr_min={args.lr_min}, args.lr_max={args.lr_max}")
    train = pd.read_csv(args.train_file)
    train = shuffle(train)
    len_train = int(train.shape[0] * 0.75)
    train_data = train[:len_train]
    validation_data = train[len_train:]

    dictionnary_labels_per_image = preprocess(train, args.train_dir)
    traindataset = MyDataset(img_id=train.image_id.values, augment_data=args.augment_data, train_images=args.train_dir, dictionnary_labels_per_image=dictionnary_labels_per_image)

    ## Pour le validation_loadr il faut que data augment soit tjr false

    validationdataset = MyDataset(validation_data.image_id.values, augment_data=args.augment_data, train_images=args.train_dir, dictionnary_labels_per_image=dictionnary_labels_per_image)
    validation_loader = torch.utils.data.DataLoader(validationdataset, batch_size=20, shuffle=True, num_workers=6,
                                                    pin_memory=True)
    ### Change number of workers to 4 and set pin_memory = True
    ### If you want a variable to not use graph tree use .detach() insted of .numpy() or .float()
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    testdataset = DatasetTest(img_id=os.listdir(args.test_dir.replace('/','')), test_dir= args.test_dir)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.leslie:
        leslie_logs = lesli_lr_range(lr_min=args.lr_min, lr_max=args.lr_max, train_loader=train_loader)
        with open('leslie_test.pickle', 'wb') as handle:
            pickle.dump(leslie_logs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = centernet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.backends.cudnn.benchmark = False
    epochs = args.epoch
    losts = []
    logs_training = train_loop(model, scheduler_name=None, epochs=epochs)
    with open('training_log.pickle', 'wb') as handle:
        pickle.dump(logs_training, handle, protocol=pickle.HIGHEST_PROTOCOL)

