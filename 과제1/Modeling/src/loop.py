from tqdm import tqdm
import torch
from sklearn.metrics import mean_absolute_error

def train_loop(dataloader, model, loss_fn, optimizer):
    #model.to('cuda')
    t_bar = tqdm(dataloader)
    losses = 0
    label_list = []
    pred_list = []
    for X, y in t_bar:
        #X = X.to('cuda')
        #y = y.to('cuda')
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        t_bar.set_description(f" MAE loss: {pow(loss,0.5):.2f}")
        losses += loss.item()

        label_list.extend(y.tolist())
        pred_list.extend(pred.tolist())
    # return mse, mae
    mae = mean_absolute_error(label_list, pred_list)
    print(f'mae : {mae :.2f}',end = ' ')
    return losses / len(dataloader), mae

def valid_loop(dataloader, model, loss_fn):
    losses = 0
    label_list = []
    pred_list = []
    with torch.no_grad():
        for X, y in dataloader:
            #X = X.to('cuda')
            #y = y.to('cuda')
            pred = model(X)
            losses += loss_fn(pred, y).item()

            label_list.extend(y.tolist())
            pred_list.extend(pred.tolist())
    mae = mean_absolute_error(label_list, pred_list)
    print(f"Validation MAE: {mae:.2f}")
    return losses / len(dataloader),mae