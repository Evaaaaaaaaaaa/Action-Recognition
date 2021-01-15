from sklearn import metrics
import scipy.io
import os
import numpy as np


# 1. Use pycharm
# 2. find the max segment length for all data to use the same model
# 3. test on different activation and loss functions
# 4. cut video with 0s at the beginning and the end of each video.
# 5. add mask at sample weight parameter when fitting
# 6. test on changing masking layer to input layer  (I am not sure if the masking layer achieves the same purpose as point 4)
# 7. add 10 activities to the training data
# 8. Form a report for all experiments
# 9. Compute result in better format

# import data
path = 'groundTruthmat/'
dataset = []
files = os.listdir(path)
data_dict = {}
for file in files:
    read = scipy.io.loadmat(path + file)
    data_dict[file] = read['labseqid']


# Helper functions
def one_hot():
    # create frame_label for one hot encoding
    frame_label = {}
    for i in range(48):
        frame_label[i] = 48 * [0]
        frame_label[i][i] = 1
    frame_label[-1] = 48 * [0]
    return frame_label


frame_label = one_hot()


def get_input_frames(data_dict, trainProportion):
    new_dict = {}
    train_y = []
    for filename, frames in data_dict.items():
        frameLen = len(frames)

        inputLen = round(frameLen * trainProportion)
        inputFrames = frames[:inputLen]
        new_dict[filename] = inputFrames
    return new_dict


def get_input_y(data_dict, input_dict):
    train_y = []
    for filename, frames in data_dict.items():
        frame_len = len(input_dict[filename])
        #         print(input_dict[filename])
        y = [input_dict[filename][frame_len - 1][0]]
        for frame in frames[frame_len:]:
            #             print(frame[0], y[-1])s
            if frame[0] != y[-1]:
                y.append(frame[0])
        y.pop(0)

        train_y.append(y)
    return train_y

# convert seccessive frmaes of same action to single action
""" data_dict structure eample
    {'P25_stereo01_P25_sandwich.mat': array([[ 0],
        [ 0],
        [ 0],
        ...,
        [39],
        [39],
        [39]], dtype=int32),"""

def frames_to_action(input_dict):
    for filename, frames in input_dict.items():
        action_list = []
        for frame in frames:
            if len(action_list) == 0:
                action_list.append(frame[0])
            else:
                if frame[0] != action_list[-1]:
                    action_list.append(frame[0])
        input_dict[filename] = action_list
    return input_dict

# add padding, each video has same length of action input
def add_padding(data_dict):
    maxLen = 0
    count = 0
    for frames in data_dict.values():
        if len(frames) > maxLen:
            maxLen = len(frames)
    for filename, frames in data_dict.items():
#         print(frames)
        data_dict[filename] = (maxLen - len(frames)) * [-1]  + frames
    return data_dict, maxLen


def add_padding_to_y(trainY):
    maxLen = 0
    count = 0
    new_trainY = []
    for frames in trainY:
        if len(frames) > maxLen:
            maxLen = len(frames)
    for frames in trainY:
        temp = (maxLen - len(frames)) * [-1] + frames
        new_trainY.append(temp)
    return new_trainY, maxLen


def feature_encoding(input_dict):
    for filename, frames in input_dict.items():
        # get corresponding one-hot encode
        new = []
        for each in frames:
            new.append(frame_label[each])
        input_dict[filename] = new
    return input_dict


def label_encoding(trainY):
    new_trainY = []
    for frames in trainY:
        new = []
        for each in frames:
            new.append(frame_label[each])
        new_trainY.append(new)
    return new_trainY

# LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, RepeatVector
from keras.layers import Dropout, Masking, TimeDistributed, Activation


def lstm_model(frame_len, max_timesteps):
    model = Sequential()
    model.add(Masking(mask_value=48 * [0], input_shape=(frame_len, 48)))
    model.add(LSTM(256))
    model.add(RepeatVector(23))
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(48)))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'], sample_weight_mode="temporal")

    model.summary()
    return model
