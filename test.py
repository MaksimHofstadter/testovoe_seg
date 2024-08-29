import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import os
ROOT_PATH = "data"
import models
import loss_utils 
import utils


def score_model(model, metric, data):
    model.eval()
    scores = 0
    for X_batch, Y_label in data:
        Y_pred = torch.sigmoid(model(X_batch.to(device))) > 0.5
        scores += metric(Y_pred, Y_label.to(device)).mean().item()

    return scores/len(data)

#Функция для построения графиков лосса и скора по эпохам
def make_graph(history, model_name, loss_name):
    fig, ax = plt.subplots(1, 2, figsize = (14, 7))
    x = history["epochs"]
    loss_train = history["train"]["loss"]
    loss_val = history["val"]["loss"]
    score_train = history["train"]["score"]
    score_val = history["val"]["score"]
    ax[0].plot(x, loss_train, label = "train", color = "green")
    ax[0].plot(x, loss_val, label = "val", color = "orange")
    ax[0].legend(fontsize = 14)
    ax[0].grid(linestyle = "--")
    ax[0].tick_params(labelsize = 14)
    ax[0].set_xlabel("epoch", fontsize = 14)
    ax[0].set_ylabel("loss", fontsize = 14)
    ax[0].set_title("Loss vs epoch", fontsize = 16)
    ax[0].set_xlim(left = 0, right = x.max())
    ax[0].set_ylim(bottom = 0)
    ax[1].plot(x, score_train, label = "train", color = "green")
    ax[1].plot(x, score_val, label = "val", color = "orange")
    ax[1].legend(fontsize = 14)
    ax[1].grid(linestyle = "--")
    ax[1].tick_params(labelsize = 14)
    ax[1].set_xlabel("epoch", fontsize = 14)
    ax[1].set_ylabel("score", fontsize = 14)
    ax[1].set_title("Score vs epoch", fontsize = 16)
    ax[1].set_xlim(left = 0, right = x.max())
    ax[1].set_ylim(bottom = 0)
    plt.suptitle(f"Model = {model_name}, loss = {loss_name}", fontsize = 18, y=1.05)
    plt.tight_layout()
    plt.show()


def scores(model, data_val, data_ts):
    val_score = score_model(model, loss_utils.iou_pytorch, data_val)
    test_score = score_model(model, loss_utils.iou_pytorch, data_ts)
    print(f"\nScore на валидации: {val_score:.4f}, score на тесте: {test_score:.4f}")
    return val_score, test_score


def train(model, optimizer, loss_fn, epochs, gen_tr, data_val, used_sheduler=True):
    X_val, Y_val = next(iter(data_val))
    history = {"epochs": np.arange(epochs)+1, "train": {"score": [], "loss": []},  "val": {"score": [], "loss": []}}
    max_train_steps = 10
    if used_sheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//3,epochs//2,epochs//1.4,epochs//1.1], gamma=0.8)

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        avg_loss = 0
        avg_score = 0
        avg_loss_val = 0
        avg_score_val = 0
        
        train_steps = 0

        model.train() 

        while train_steps < max_train_steps:
        	X_train, Y_train = next(gen_tr)
	        data_tr = DataLoader(list(zip(X_train, Y_train)), 
		                     batch_size=16, 
		                     shuffle=True, 
		                     num_workers=1)
	        
	        X_batch, Y_batch = next(iter(data_tr))
	        
	        X_batch = X_batch.to(device)
	        Y_batch = Y_batch.to(device)

	        optimizer.zero_grad()

	            # forward
	        Y_pred = model(X_batch)
	        loss =  loss_fn(Y_pred, Y_batch)
	        loss.backward()
	        optimizer.step() # update weights
	        score = loss_utils.iou_pytorch(torch.sigmoid(Y_pred) > 0.5, Y_batch).mean().item()

	        avg_loss += loss / max_train_steps
	        avg_score += score/ max_train_steps
	        train_steps = train_steps+1

        print('loss: %f' % avg_loss)
        if used_sheduler:
          scheduler.step()

        model.eval()
        Y_hat = model(X_val.to(device)).detach().to("cpu")
        loss_val = loss

        # Visualize tools
        for Xv_batch, Yv_batch in data_val:
            print("train, ", Xv_batch.shape)
            Xv_batch = Xv_batch.to(device)
            Yv_batch = Yv_batch.to(device)
            Y_pred_val = model(Xv_batch)
            loss_val = loss_fn(Y_pred_val, Yv_batch)
            score_val = loss_utils.iou_pytorch(torch.sigmoid(Y_pred_val) > 0.5, Yv_batch).mean().item()
            avg_loss_val += loss / len(data_val)
            avg_score_val += score_val/ len(data_val)
            
        history["train"]["score"].append(avg_score)
        history["val"]["score"].append(avg_score_val)
        history["train"]["loss"].append(avg_loss.item())
        history["val"]["loss"].append(avg_loss_val.item())
        fig, ax = plt.subplots(3, 6, figsize = (14, 12))
        
        for k in range(6):
            ax[0, k].imshow(np.rollaxis(X_val[k].numpy(), 0, 3), cmap='gray')
            ax[0, k].set_title("Real")
            ax[0, k].axis('off')
            ax[1, k].imshow(torch.sigmoid(Y_hat[k, 0]) > 0.5, cmap='gray')
            ax[1, k].set_title("Output")
            ax[1, k].axis('off') 
            ax[2, k].imshow(Y_val[k, 0], cmap='gray')
            ax[2, k].set_title("Ground Truth")
            ax[2, k].axis('off')                    
        plt.suptitle('%d / %d - train_loss: %f , val_loss: %f, train_score: %f, val_score: %f' % (epoch+1, epochs, avg_loss, avg_loss_val, avg_score, avg_score_val))
        plt.tight_layout()
        plt.show()
        
    # очистка кеша

    X_batch.to("cpu")
    Y_batch.to("cpu")
    Xv_batch.to("cpu")
    Yv_batch.to("cpu")
    del model
    del X_batch
    del Y_batch
    del Xv_batch
    del Yv_batch
    torch.cuda.empty_cache()

    return history

if __name__ == '__main__': 
	
	MAX_EPOCHS = 1
	BASE_LR = 3e-3
	WEIGHT_DECAY = 0.01
	batch_size = 16
	size = (256, 256)


	images_list_val = []
	mask_list_val = []

	for root, dirs, files in os.walk(ROOT_PATH+"/val"):
		for i in files:
			images_list_val.append(imread(ROOT_PATH+"/val/"+ i))
			mask_list_val.append(imread(ROOT_PATH+"/val_pred/"+ i.split(".")[0]+ ".bmp"))

	images_list_ts = []
	mask_list_ts = []
	for root, dirs, files in os.walk(ROOT_PATH+"/ts"):
		for i in files:
			images_list_ts.append(imread(ROOT_PATH+"/ts/"+ i))
			mask_list_ts.append(imread(ROOT_PATH+"/ts_pred/"+ i.split(".")[0]+ ".bmp"))

	
	print(f'Loaded images')

	
	X_val = [resize(x, size, mode='constant', anti_aliasing=True,) for x in images_list_val]
	Y_val = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in mask_list_val]
	X_val = np.array(X_val, np.float32)
	Y_val = np.array(Y_val, np.float32)

	X_ts = [resize(x, size, mode='constant', anti_aliasing=True,) for x in images_list_ts]
	Y_ts = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in mask_list_ts]
	X_ts = np.array(X_ts, np.float32)
	Y_ts = np.array(Y_ts, np.float32)
	

	data_gen = utils.generator_train_batch(batch_size, size, ROOT_PATH)
	data_val = DataLoader(list(zip(np.rollaxis(X_val, 3, 1), Y_val[:,np.newaxis])),
	                      batch_size=batch_size, shuffle=False, 
	                      num_workers=2)
	data_ts = DataLoader(list(zip(np.rollaxis(X_ts, 3, 1), Y_ts[:,np.newaxis])),
	                     batch_size=batch_size, shuffle=False, 
	                     num_workers=2)


	train_on_gpu = torch.cuda.is_available()

	if not train_on_gpu:
	    print('CUDA is not available.  Training on CPU ...')
	else:
	    print('CUDA is available!  Training on GPU ...')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print(device)
	

	model = models.UNet().to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr = BASE_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY)
	h_segnet_bce = train(model, optimizer, loss_utils.dice_loss, MAX_EPOCHS, data_gen, data_val, used_sheduler=True)
	torch.save(model.state_dict(), 'my_model_dice.pth')

	print(f'Save model')

	segnet_bce_val_score, segnet_bce_test_score = scores(model, data_val, data_ts)

	make_graph(h_segnet_bce, "UNet", "DICE")
	
