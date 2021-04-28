import torch.optim as optim
from tqdm import tqdm
import gc
from model import *
from loss_function import *


def lesli_lr_range(lr_min, lr_max, train_loader, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = centernet()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=((1e-2)-(1e-5))/30)
    logs = []
    model.train()

    running_loss = 0.0
    running_mask = 0.0
    running_regr = 0.0
    running_offset = 0.0

    for epoch in range(epochs):
        t = tqdm(train_loader)
        for idx, (img, index, hm, regr, offset) in enumerate(t):
            # send to gpu
            img = img.to(device)
            hm_gt = hm.to(device)
            regr_gt = regr.to(device)
            offset_gt = offset.to(device)


            optimizer.zero_grad(set_to_none=True)

            # run model
            hm, regr, offset = model(img)
            preds = torch.cat((hm, regr, offset), 1)

            loss, mask_loss, regr_loss, offset_loss = centerloss(preds, hm_gt, regr_gt, offset_gt)
            # misc
            running_loss = loss
            running_mask = mask_loss
            running_regr = regr_loss
            running_offset = offset_loss

            loss.backward()
            optimizer.step()

            t.set_description(
                f't (l={running_loss / (idx + 1):.3f})(m={running_mask / (idx + 1):.4f})(r={running_regr / (idx + 1):.4f})(o={running_offset / (idx + 1):.4f})')
            # scheduler.step()
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] + ((lr_max) - (lr_min)) / (181 * epochs)
            log_mini_batch = {'lr': optimizer.state_dict()['param_groups'][0]['lr'],
                              'loss': running_loss, "mask": running_mask,
                              "regr": running_regr, 'offset': running_offset}

            logs.append(log_mini_batch)
            # print("Learning rate ", optimizer.state_dict()['param_groups'][0]['lr'])
            gc.collect()
    return logs
