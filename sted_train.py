import models
import datasets
import utils
import trainer
import torch.optim as optim
import numpy as np
import torch
import pandas as pd
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('-model_save_path', help='Path to save the encoder and decoder models')

args = parser.parse_args()

batch_size = 64
learning_rate = 1e-3
weight_decay = 0
num_workers = 0 #
num_epochs = 40 #
layers_enc = 2
layers_dec = 2
dropout_p = 0 #
num_hidden = 256
normalize = True
device = torch.device("cuda")
model_save_path = args.model_save_path

encoder = models.EncoderRNN(device, num_hidden, layers_enc)
encoder = encoder.to(device)
encoder = encoder.float()
decoder = models.DecoderRNN(device, num_hidden, dropout_p, layers_dec)
decoder = decoder.to(device)
decoder = decoder.float()

try:
    train_boxes = np.load('/home/u3465097/0711_MOF/all0913/add_radar/train_box_statistics_final_radar_bonus.npy') #train_box_statistics_final.npy') #add_radar/train_box_statistics_final_radar_bonus.npy') #/home/u3465097/0711_MOF/all0913/train_box_statistics_final.npy') #train_box_velo_lowest_depth_rawRadar_statistics.npy')
    train_labels = np.load('/home/u3465097/0711_MOF/all0913/add_radar/train_targets_final_radar.npy') #train_targets_final.npy') #add_radar/train_targets_final_radar.npy') #/home/u3465097/0711_MOF/all0913/train_targets_final.npy') #train_targets.npy') # np.load('train_targets.npy')
    val_boxes = np.load('/home/u3465097/0711_MOF/all0913/add_radar/valid_box_statistics_final_radar_bonus.npy') #valid_box_statistics_final.npy') #add_radar/valid_box_statistics_final_radar_bonus.npy') #/home/u3465097/0711_MOF/all0913/valid_box_statistics_final.npy') #valid_box_velo_lowest_depth_rawRadar_statistics.npy')
    val_labels = np.load('/home/u3465097/0711_MOF/all0913/add_radar/valid_targets_final_radar.npy') #valid_targets_final.npy') #add_radar/valid_targets_final_radar.npy') #/home/u3465097/0711_MOF/all0913/valid_targets_final.npy') #valid_targets.npy')

except Exception:
    print('Failed to load data') #from ' + str(data_path))
    exit()

print(train_boxes.shape)
print(train_labels.shape)
print(val_boxes.shape)
print(val_labels.shape)

# Normalize boxes
for i in range(10):
    val_boxes[:, i, ] = (val_boxes[:, i, ] - train_boxes[:, i, ].mean()) / \
        train_boxes[:, i, ].std()
    train_boxes[:, i, ] = (train_boxes[:, i, ] - train_boxes[:,
                                                             i, ].mean()) / train_boxes[:, i, ].std()

loss_function = torch.nn.SmoothL1Loss()

train_set = datasets.Simple_BB_Dataset(train_boxes, train_labels) #, train_dtp_features)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_set = datasets.Simple_BB_Dataset(val_boxes, val_labels) #, val_dtp_features)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

optimizer_encoder = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_decoder = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

best_ade = np.inf
for epoch in range(num_epochs):
    print('----------- EPOCH ' + str(epoch+1) + ' -----------')
    print('Training...')
    trainer.train_seqseq(encoder, decoder, device, train_loader, optimizer_encoder, optimizer_decoder,
                         epoch, loss_function, learning_rate)
    print('Validating...')
    val_predictions, val_targets, val_ade, val_fde = trainer.test_seqseq(
        encoder, decoder, device, val_loader, loss_function, return_predictions=True)
    if epoch == 9:
        print('Adjust lr to 1e-4')
        optimizer_encoder = optim.Adam(encoder.parameters(), lr=1e-4, weight_decay=weight_decay)
        optimizer_decoder = optim.Adam(decoder.parameters(), lr=1e-4, weight_decay=weight_decay)
    if epoch == 19:
        print('Adjust lr to 5e-5')
        optimizer_encoder = optim.Adam(encoder.parameters(), lr=5e-5, weight_decay=weight_decay)
        optimizer_decoder = optim.Adam(decoder.parameters(), lr=5e-5, weight_decay=weight_decay)
    if epoch == 29:
        print('Adjust lr to 2.5e-5')
        optimizer_encoder = optim.Adam(encoder.parameters(), lr=2.5e-5, weight_decay=weight_decay)
        optimizer_decoder = optim.Adam(decoder.parameters(), lr=2.5e-5, weight_decay=weight_decay)
    if val_ade < best_ade:
        best_encoder, best_decoder = copy.deepcopy(encoder), copy.deepcopy(decoder)
        best_ade = val_ade
        best_fde = val_fde
    print('Best validation ADE: ', np.round(best_ade, 1))
    print('Best validation FDE: ', np.round(best_fde, 1))

print('Saving model weights to ', model_save_path)
torch.save(best_encoder.state_dict(), model_save_path + '/encoder_gru.weights')
torch.save(best_decoder.state_dict(), model_save_path + '/decoder_gru.weights')
