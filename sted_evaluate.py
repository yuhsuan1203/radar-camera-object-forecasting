import models
import datasets
import utils
import trainer
import numpy as np
import torch
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('-model_path', help='Path to encoder and decoder models')
#parser.add_argument('-data_path', help='Path to bounding box statistics and precomputed DTP features')

#args = parser.parse_args()

batch_size = 64
num_workers = 0 #8
layers_enc = 2
layers_dec = 2
dropout_p = 0
num_hidden = 256

device = torch.device("cuda")
#model_path = args.model_path
#data_path = args.data_path

#for detector in ['yolo', 'mask-rcnn']:
#    for fold in [1, 2, 3]:

#        print(detector + ' fold ' + str(fold))

print('loading model')

encoder = models.EncoderRNN(device, num_hidden, layers_enc)
encoder = encoder.to(device)
encoder = encoder.float()
decoder = models.DecoderRNN(device, num_hidden, dropout_p, layers_dec)
decoder = decoder.to(device)
decoder = decoder.float()

try:
    encoder_path = '/home/u3465097/0711_MOF/0719/velo_09/encoder_gru.weights'
    decoder_path = '/home/u3465097/0711_MOF/0719/velo_09/decoder_gru.weights'
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
except Exception:
    print('Failed to load model')
    exit()

encoder.eval()
decoder.eval()

#path = data_path + detector + '_features/fold' + str(fold) + '/'

print('Loading data')

try:
    train_boxes = np.load('/home/u3465097/0711_MOF/all0913/add_radar/train_box_statistics_final_radar_bonus.npy') #train_box_statistics_final.npy') #add_radar/train_box_statistics_final_radar_bonus.npy') # train_box_statistics_final.npy') # 0711_MOF/all0913/add_radar/train_box_statistics_final_radar_bonus.npy') #/home/u3465097/0711_MOF/all0913/train_box_statistics.npy')
    test_boxes = np.load('/home/u3465097/0711_MOF/all0913/add_radar/nms/test_box_statistics_radar_bonus.npy') #test_box_statistics.npy') #/home/u3465097/whole_arch/pred_scripts/test_nms_new0913.npy') #0711_MOF/all0913/test_box_statistics.npy')
    test_labels = np.load('/home/u3465097/0711_MOF/all0913/add_radar/nms/test_targets.npy') #/home/u3465097/whole_arch/pred_scripts/test_targets_new.npy') #0711_MOF/all0913/test_targets.npy')

except Exception:
    print('Failed to load data')
    exit()

#train_boxes = train_boxes[:, :8, :]
print(train_boxes.shape)
print(test_boxes.shape)
print(test_labels.shape)
#test_boxes = test_boxes[:, :8, :]

# Normalize boxes
for i in range(10):
    test_boxes[:, i, ] = (test_boxes[:, i, ] - train_boxes[:, i, ].mean()) / \
        train_boxes[:, i, ].std()
    train_boxes[:, i, ] = (train_boxes[:, i, ] - train_boxes[:,
                                                             i, ].mean()) / train_boxes[:, i, ].std()

loss_function = torch.nn.SmoothL1Loss()

test_set = datasets.Simple_BB_Dataset(
    test_boxes, test_labels) #, test_dtp_features)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers)

print('Getting predictions')

predictions, targets, ade, fde = trainer.test_seqseq(
    encoder, decoder, device, test_loader, loss_function, return_predictions=True, phase='Test')

print('Getting IOU metrics')

# Predictions are relative to constant velocity. To compute AIOU / FIOU, we need the constant velocity predictions.
test_cv_preds = pd.read_csv('/home/u3465097/0711_MOF/all0913/add_radar/nms/test_cv_negfloat.csv') #/home/u3465097/whole_arch/pred_scripts/test_cv_new_neg_float.csv') #0711_MOF/all0913/test_cv_neg_float.csv')
results_df = pd.DataFrame()
results_df['vid'] = test_cv_preds['vid'].copy()
#results_df['filename'] = test_cv_preds['filename'].copy()
#results_df['frame_num'] = test_cv_preds['frame_num'].copy()

# First 3 columns are file info. Remaining columns are bounding boxes.
test_cv_preds = test_cv_preds.iloc[:, 1:].values.reshape(len(test_cv_preds), -1, 4)
predictions = predictions.reshape(-1, 96, order='A')
predictions = predictions.reshape(-1, 24, 4)

predictions = utils.xywh_to_x1y1x2y2(predictions)
#predictions = np.swapaxes(predictions, 1, 2)

predictions = test_cv_preds - predictions #predictions = np.around(predictions).astype(int)

predictions = np.around(predictions).astype(int) #predictions = test_cv_preds - predictions
print(predictions.shape)
#print(len(predictions))
#print(len(predictions[0]))
#print(len(predictions[0][0]))
good_predictions = np.zeros(predictions.shape)

for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        for k in range(len(predictions[i][j])):
            if k % 2 == 0:
                # x coord
                good_predictions[i][j][k] = min(1600, max(0, predictions[i][j][k]))
                #if predictions[i][j][k] < 0 or predictions[i][j][k] > 1600:
                #    print(i, j, k, 'error_x')
            else:
                good_predictions[i][j][k] = min(900, max(0, predictions[i][j][k]))
                #if predictions[i][j][k] < 0 or predictions[i][j][k] > 900:
                #    print(i, j, k, 'error_y')


for i in range(len(good_predictions)):
    for j in range(len(good_predictions[i])):
        for k in range(len(good_predictions[i][j])):
            if k % 2 == 0:
                # x coord
                if good_predictions[i][j][k] < 0 or good_predictions[i][j][k] > 1600:
                    print(i, j, k, 'error_x')
            else:
                if good_predictions[i][j][k] < 0 or good_predictions[i][j][k] > 900:
                    print(i, j, k, 'error_y')

gt_df = pd.read_csv('/home/u3465097/0711_MOF/all0913/add_radar/nms/test_gt.csv') #/home/u3465097/whole_arch/pred_scripts/test_gt_new.csv') #0711_MOF/all0913/test_gt.csv')
gt_boxes = gt_df.iloc[:, 1:].values.reshape(len(gt_df), -1, 4)
print('pred:', good_predictions.shape)
print('gt:', gt_boxes.shape)
aiou = utils.calc_aiou(gt_boxes, good_predictions)
fiou = utils.calc_fiou(gt_boxes, good_predictions)
print('AIOU: ', round(aiou * 100, 1))
print('FIOU: ', round(fiou * 100, 1))

print('Saving predictions')

for i in range(1, 25):
    results_df['x1_' + str(i)] = good_predictions[:, i - 1, 0]
    results_df['y1_' + str(i)] = good_predictions[:, i - 1, 1]
    results_df['x2_' + str(i)] = good_predictions[:, i - 1, 2]
    results_df['y2_' + str(i)] = good_predictions[:, i - 1, 3]

results_df.to_csv('/home/u3465097/0711_MOF/all0913/add_radar/nms/test_pred_velo_09.csv', index=False) #/home/u3465097/whole_arch/pred_scripts/new_pred_results/test_pred_no_velo_01.csv', index=False) #0711_MOF/all0913/test_pred_no_velo_01.csv', index=False)
