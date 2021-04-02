import pandas as pd
import torch.optim as optim
from tqdm import tqdm
import gc
import os
from args_parse import *
from preprocess import *
from model import *
from dataloaders import *
from loss_function import *

args = parse_args()

def train_model(epoch):
    model.train()
    print('epochs {}/{} '.format(epoch + 1, epochs))
    running_loss = 0.0
    running_mask = 0.0
    running_regr = 0.0
    t = tqdm(train_loader)
    rd = np.random.rand()

    for idx, (img, index, hm, regr) in enumerate(t):
        # send to gpu
        img = img.to(device)
        hm_gt = hm.to(device)
        regr_gt = regr.to(device)
        # set opt
        ## set_to_none=True apparently optimizes the memory usage
        optimizer.zero_grad(set_to_none=True)

        # run model
        hm, regr = model(img)
        preds = torch.cat((hm, regr), 1)

        loss, mask_loss, regr_loss = centerloss(preds, hm_gt, regr_gt)
        # misc
        running_loss += loss
        running_mask += mask_loss
        running_regr += regr_loss

        loss.backward()
        optimizer.step()

        t.set_description(f"t (l={running_loss / (idx + 1):.3f})(m={running_mask / (idx + 1):.4f})(r={running_regr / (idx + 1):.4f})")

    # scheduler.step()
    print('train loss : {:.4f}'.format(running_loss / len(train_loader)))
    print('maskloss : {:.4f}'.format(running_mask / (len(train_loader))))
    print('regrloss : {:.4f}'.format(running_regr / (len(train_loader))))

    # save logs
    log_epoch = {'epoch': epoch + 1, 'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                 'loss': running_loss / len(train_loader), "mask": running_mask / (len(train_loader)),
                 "regr": running_regr / (len(train_loader))}
    logs.append(log_epoch)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("main lancé")
    train = pd.read_csv(args.train_file)
    print(f"args.epoch = {args.epoch}, args.lr={args.lr}, args.batch_size={args.batch_size}, args.num_workers={args.num_workers}\n"
          f"args.train_file={args.train_file}, args.train_dir={args.train_dir}, args.test_dir={args.test_dir}")
    dictionnary_labels_per_image = preprocess(train, args.train_dir)
    traindataset = MyDataset(train.image_id.values, dictionnary_labels_per_image, args.train_dir)
    ### Change number of workers to 4 and set pin_memory = True
    ### If you want a variable to not use graph tree use .detach() insted of .numpy() or .float()
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    testdataset = DatasetTest(os.listdir(args.test_dir.replace('/','')), test_dir= args.test_dir)
    test_loader = torch.utils.data.DataLoader(testdataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = centernet()
    model.to(device)
    ### C'est supposé rendre le modele plus rapide mais ca ne semble pas fonctionner
    torch.backends.cudnn.benchmark = False
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epoch
    logs = []
    logs_eval = []
    TRAIN = True
    if TRAIN:
        for epoch in range(epochs):
            train_model(epoch)
            torch.save(model.state_dict(), "model")
            gc.collect()
    else:
        model.load_state_dict(torch.load("model"))
    ### Around 40 epochs runned
    ### Mask loss fell to 0.1041 and regrloss fell to 6.4058 and train loss to 6.5094
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
