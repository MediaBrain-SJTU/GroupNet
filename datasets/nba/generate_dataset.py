
from Game import Game
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Process arguments about an NBA game.')

args = parser.parse_args()

data_root = 'datasets/source'
data_target = 'datasets'
if not os.path.exists(data_target):
    os.mkdir(data_target)
json_list = os.listdir(data_root)
print(json_list)
all_trajs = []

for file_name in json_list:
	if '.json' not in file_name:
		continue
	json_path = data_root + '/' + file_name
	game = Game(path_to_json=json_path)
	trajs = game.read_json()   
	trajs = np.unique(trajs,axis=0) 
	print(trajs.shape)
	all_trajs.append(trajs)

all_trajs = np.concatenate(all_trajs,axis=0)
all_trajs = np.unique(all_trajs,axis=0)
print(len(all_trajs))
index = list(range(len(all_trajs)))
from random import shuffle
shuffle(index)
train_set = all_trajs[index[:37500]]
test_set = all_trajs[index[37500:]]
print('train num:',train_set.shape[0])
print('test num:',test_set.shape[0])

np.save(data_target+'/train.npy',train_set)
np.save(data_target+'/test.npy',test_set)


# data_root_train = '/DATA7_DB7/data/cxxu/NBA-Player-Movements/data_subset/subset_1/train.npy'
# data_root_test = '/DATA7_DB7/data/cxxu/NBA-Player-Movements/data_subset/subset_1/test.npy'

# train_data = np.load(data_root_train)
# test_data = np.load(data_root_test)

# train_data_new = train_data[:37500]
# test_data_new = np.concatenate((test_data,train_data[37500:]),axis=0)
# print(train_data_new.shape)
# print(test_data_new.shape)

# a = [35, 66, 101, 196, 436, 500, 582, 910, 1285, 1334, 1353, 1376, 1520, 1549, 1585, 1639, 1653, 1768, 1929, 1975, 2046,
#  2081, 2136, 2196, 2272, 2432, 2475, 2796, 2844, 2865, 2956, 3275, 3328, 3357, 3371, 3717, 3740, 4226, 4227, 4232, 
# 4378, 4454, 4588, 5013, 5035, 5095, 5210, 5534, 5590, 5688, 5699, 5812, 6021, 6157, 6257, 6493, 6640, 6646, 6752, 
# 7085, 7113, 7120, 7272, 7304, 7352, 7438, 7483, 7648, 7855, 7937, 7978, 8044, 8053, 8099, 8147, 8181, 8217, 8238, 8248,
#  8352, 8498, 8840, 8910, 9156, 9171, 9193, 9218, 9317, 9843, 10041, 10241, 10332, 10358, 10396, 10482, 10580, 10614, 10849, 10876, 10894, 10921, 10932, 10944, 10958, 11367, 11563, 11624, 11786, 12033, 12075, 12455, 12464, 12502, 
# 12613, 12678, 12718]
# # a = [35, 66, 99, 490, 571, 897, 1271, 1338, 1746, 1906, 1951, 2022, 2057, 2246, 2406, 2449, 2769, 2817, 2837, 2925, 3244, 3297, 3339, 3682, 3705, 4190, 4339, 4546, 4971, 5052, 5166, 5640, 5651, 5764, 5973, 6109, 6444, 7065, 7216, 7248, 7296, 7425, 7590, 7797, 7879, 7920, 7986, 8120, 8156, 8186, 8289, 8845, 9105, 9771, 10257, 10283, 10321, 10406, 10502, 10535, 10796, 10840, 10851, 10876, 11285, 11702, 11990, 12375]
# # a = [92, 95, 196, 251, 302, 307, 323, 341, 428, 436, 580, 692, 837, 1101, 1334, 1376, 1520, 1549, 1585, 1639, 1653, 1663, 1884, 1958, 2136, 2196, 2789, 2861, 2906, 2951, 2953, 3357, 3486, 3541, 3595, 3925, 4227, 4232, 4271, 4439, 4451, 4454, 5035, 5185, 5284, 5292, 5534, 5590, 6257, 6640, 6646, 6752, 7056, 7085, 7113, 7235, 7382, 7438, 8053, 8099, 8147, 8238, 8325, 8498, 8840, 9156, 9193, 9196, 9218, 9317, 9581, 9799, 10041, 10241, 10324, 10478, 10490, 10541, 10594, 10849, 10894, 10944, 11563, 11624, 12033, 12183, 12256, 12451, 12455, 12502, 12605, 12630, 12736]
# print(len(a))
# test_data_new = np.delete(test_data_new,a,axis=0) 
# print(test_data_new.shape)

# np.save(data_target+'/train.npy',train_data_new)
# np.save(data_target+'/test.npy',test_data_new)