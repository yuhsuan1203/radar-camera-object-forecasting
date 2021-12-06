import argparse
import pandas as pd
import utils
import numpy as np

box_names = []

for i in range(1, 25):
    box_names.append('x1_' + str(i))
    box_names.append('y1_' + str(i))
    box_names.append('x2_' + str(i))
    box_names.append('y2_' + str(i))

#parser = argparse.ArgumentParser()
#parser.add_argument('-gt', help='Path to ground truth directory')
#parser.add_argument('-pred', help='Path to prediction directory')
#args = parser.parse_args()

#for detector in ['yolo', 'mask-rcnn']:
#aious = []
#fious = []
#ades = []
#fdes = []
#for fold in [1, 2, 3]:

gt_df = pd.read_csv('/home/u3465097/0711_MOF/all0913/add_radar/nms/test_gt.csv') #/home/u3465097/0711_MOF/all0913/add_radar/test_gt_final_radar.csv') #/home/u3465097/whole_arch/pred_scripts/old_csvfiles/test_gt_nms.csv')
pred_df = pd.read_csv('/home/u3465097/0711_MOF/all0913/add_radar/nms/test_cv.csv') #/home/u3465097/0711_MOF/all0913/add_radar/test_pred_final_velo_02_bonus.csv') #/home/u3465097/whole_arch/pred_scripts/old_csvfiles/test_pred_models_nvnc03_e2d2_nms01.csv') #test_cv.csv') # test_cv.csv')
#gt_df = pd.read_csv('/home/u3465097/0711_MOF/all0913/test_gt.csv')
#pred_df = pd.read_csv('/home/u3465097/0711_MOF/all0913/test_pred_no_velo_01.csv') #test_cv.csv') # test_cv.csv')

gt_boxes = gt_df[box_names].values.reshape(len(gt_df), -1, 4)
pred_boxes = pred_df[box_names].values.reshape(len(pred_df), -1, 4)

ade = utils.calc_ade(gt_boxes, pred_boxes) #, return_mean=False)
fde = utils.calc_fde(gt_boxes, pred_boxes)
aiou = utils.calc_aiou(gt_boxes, pred_boxes)
fiou = utils.calc_fiou(gt_boxes, pred_boxes)

#aious.append(aiou)
#fious.append(fiou)
#ades.append(ade)
#fdes.append(fde)

#print('test_pred_models_wvnc03_e2d2.csv:')
#print(detector + ' fold ' + str(fold))
print('AIOU: ', round(aiou * 100, 1))
print('FIOU: ', round(fiou * 100, 1))
#print('ADE:  ', np.mean(ade))
print('ADE:  ', round(ade, 1))
#print('FDE:  ', fde)
print('FDE:  ', round(fde, 1))
#print(ade.shape)
#print(ade)

#print()
#print('Mean over 3 folds for ' + detector)
#print('AIOU: ', round(np.mean(aious) * 100, 1))
#print('FIOU: ', round(np.mean(fious) * 100, 1))
#print('ADE:  ', round(np.mean(ades), 1))
#print('FDE:  ', round(np.mean(fdes), 1))
