import torch.optim as optim
from tqdm import tqdm
import gc
from model import *
from loss_function import *
import matplotlib.pyplot as plt


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

def plot_leslie(logs, file_name):
    lr = []
    loss = []
    loss_reg = []
    loss_mask = []
    loss_offset = []
    for l in logs:
      lr.append(l['lr'])
      loss.append(l['loss'])
      loss_reg.append(l['regr'])
      loss_mask.append(l['mask'])
      loss_offset.append(l['offset'])

    beta = 0.8
    avg_loss = 0
    avg_loss_regr = 0
    avg_loss_mask = 0
    avg_loss_offset = 0
    smoothed_losses=[]
    smoothed_losses_regr=[]
    smoothed_losses_mask=[]
    smoothed_losses_offset=[]

    for i,l in enumerate(loss):
      avg_loss = beta * avg_loss + (1-beta) *l
      smoothed_loss = avg_loss / (1 - beta**i)
      smoothed_losses.append(smoothed_loss)

    for i,l in enumerate(loss_reg):
      avg_loss_regr = beta * avg_loss_regr + (1-beta) *l
      smoothed_loss_regr = avg_loss_regr / (1 - beta**i)
      smoothed_losses_regr.append(smoothed_loss_regr)

    for i,l in enumerate(loss_mask):
      avg_loss_mask = beta * avg_loss_mask + (1-beta) *l
      smoothed_loss_mask = avg_loss_mask / (1 - beta**i)
      smoothed_losses_mask.append(smoothed_loss_mask)

    for i,l in enumerate(loss_offset):
      avg_loss_offset = beta * avg_loss_offset + (1-beta) *l
      smoothed_loss_offset = avg_loss_offset / (1 - beta**i)
      smoothed_losses_offset.append(smoothed_loss_offset)

    fig, axs = plt.subplots(4, 2)
    axs[0, 0].plot(lr, smoothed_losses_regr)
    axs[0, 0].set_title('smoothed_losses_regr')
    axs[0, 1].plot(lr[:100], smoothed_losses_regr[:100])
    axs[0, 1].set_title('100 first smoothed_losses_regr')
    axs[1, 0].plot(lr, smoothed_losses)
    axs[1, 0].set_title('smoothed_losses')
    axs[1, 1].plot(lr[:100], smoothed_losses[:100])
    axs[1, 1].set_title('100 first smoothed_losses')
    axs[2, 0].plot(lr, smoothed_losses_mask)
    axs[2, 0].set_title('smoothed_losses_mask')
    axs[2, 1].plot(lr[:100], smoothed_losses_mask[:100])
    axs[2, 1].set_title('100 first smoothed_losses_mask')
    axs[3, 0].plot(lr, smoothed_losses_offset)
    axs[3, 0].set_title('smoothed_losses_offset')
    axs[3, 1].plot(lr[:100], smoothed_losses_offset[:100])
    axs[3, 1].set_title('100 first smoothed_losses_offset')
    fig.tight_layout()
    fig.savefig("Leslie_lr_"+file_name+".png")
