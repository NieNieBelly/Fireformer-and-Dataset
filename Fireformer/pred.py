import scipy.io
import pandas as pd
import torch.cuda
import h5py
import netCDF4 as nc
import numpy as np
import time
from myNet.Fireformer import Fireformer
from compareNet.SMCNN import SMCNN
from compareNet.EfficientNet import EfficientNetV2
from compareNet.STSRNN import STSRNN
from utils.sequenceDataset import TestDataSet, TestDataSet1
from utils.makeSequenceData import split37File
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.io as scio


def getMask(h5_path):
    h5_file = h5py.File(h5_path, "r")
    fire_mask = np.asarray(h5_file["firespot"])
    cloud_mask = np.asarray(h5_file["cloud_mask"])
    water_mask = np.asarray(h5_file["water_mask"])
    h5_file.close()
    return fire_mask, cloud_mask, water_mask


def getNcData(nc_path, keywords):
    dataset = nc.Dataset(nc_path)
    data = []
    for keyword in keywords:
        data.append(dataset.variables[keyword])
    data = np.asarray(data)
    return data


def data2img(data, mask, save_name, save=True, color='yellow', split_range=True):
    image_arr = []
    if split_range:
        _ = data[0, :, :]
        _range = np.max(_) - np.min(_)
        _ = ((_ - np.min(_)) / _range) * 255
        image_arr.append(_)
        _ = data[1, :, :]
        _range = np.max(_) - np.min(_)
        _ = ((_ - np.min(_)) / _range) * 255
        image_arr.append(_)
        _ = data[2, :, :]
        _range = np.max(_) - np.min(_)
        _ = ((_ - np.min(_)) / _range) * 255
        image_arr.append(_)
    else:
        image_arr = data

    # 标准化
    image_arr = np.asarray(image_arr)
    _range = np.max(image_arr) - np.min(image_arr)
    image_arr = ((image_arr - np.min(image_arr)) / _range) * 255
    image_arr = np.asarray(image_arr, dtype=int)
    image_arr = np.transpose(image_arr, (1, 2, 0))

    if save:
        new_im = Image.fromarray(np.uint8(image_arr))
        new_im.save(save_name[:-9] + ".jpg")

    plt.imshow(image_arr)
    pred_index = np.where(mask == 1)
    pred_index = list(pred_index)
    pred_point = list(zip(pred_index[0], pred_index[1]))
    for point in pred_point:
        plt.scatter(point[1], point[0], s=10, c='', edgecolors=color, marker="s")
    # plt.show()
    plt.savefig(save_name)
    plt.clf()


def saveRes(data, savename):
    scio.savemat(savename, {"pre": data})


if __name__ == "__main__":
    all_cost_time = 0.0
    DAY = 7
    BS = 500
    keywords = ["B03", "B04", "B06", "B07", "B14", "B15"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc_folder = "E:/dierlunquanbushuju/2021"
    Fire_P = 0.8811
    provinces = ["JX"]

    model_path = "D:/ld/temp_model/Fireformer_20250110_1200_CP_epoch200.pth"
    net = Fireformer()

    net.to(device)
    net.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    TP_, TN_, FP_, FN_ = 0.0, 0.0, 0.0, 0.0
    img_num = 0
    T0300, T0700 = split37File(nc_folder)
    T0700 = sorted(T0700, reverse=True)
    T0300 = sorted(T0300, reverse=True)
    for num_seq_0700 in range(len(T0700) - DAY + 1):
        for province in provinces:
            time_seq = T0700[num_seq_0700:num_seq_0700 + DAY]
            h5_path = f"{nc_folder}/{time_seq[0]}/{province}/debug.h5"
            if not os.path.exists(h5_path):
                continue
            nc_paths = [f"{nc_folder}/{time_}/{province}/grid_hw08_{province}_2km_{time_}.nc" for time_ in time_seq]
            firemask, cloudmask, watermask = getMask(h5_path)
            if np.sum(firemask) <= 0:
                continue
            dataset = TestDataSet(nc_paths, 21, keywords)
            if len(dataset) <= 0:
                continue
            img_num += 1
            loader = DataLoader(dataset, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
            pred = []
            start_time = time.time()
            for batch in loader:
                patchs = batch["patch"]
                patchs = patchs.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    labels_pred = net(patchs)
                    sm = torch.nn.Softmax()
                    labels_pred = sm(labels_pred)
                    pred_classes = torch.where(labels_pred > Fire_P, 1, 0)
                    pred_classes = torch.max(pred_classes, dim=1)[1]
                    pred_classes = pred_classes.cpu().numpy()
                    for pred_class in pred_classes:
                        pred.append(pred_class)
            end_time = time.time()
            cost_time = end_time - start_time
            all_cost_time += cost_time
            pred = np.asarray(pred)
            pred = np.reshape(pred, (firemask.shape[0], firemask.shape[1]))
            pred = np.where((pred == 1), 1, 0)
            # saveRes(pred, f"D:/ld/Fire-detection-result/Fireformer/{province}_{time_seq[0]}.mat")
            sub = firemask - pred
            add = firemask + pred
            TP = np.sum(np.where(add == 2, 1, 0))
            TN = np.sum(np.where(add == 0, 1, 0))
            FP = np.sum(np.where(sub == -1, 1, 0))
            FN = np.sum(np.where(sub == 1, 1, 0))
            acc = (TP + TN) / (TP + FP + TN + FN)
            recall = TP / (TP + FN)
            P = TP / (TP + FP)
            TP_ += TP
            TN_ += TN
            FP_ += FP
            FN_ += FN
            print(time_seq)
            print("TP, TN, FP, FN", TP, TN, FP, FN)
            print("Acc:", acc)
            print("Recall", recall)
            print("P", P)


    for num_seq_0300 in range(len(T0300) - DAY + 1):
        for province in provinces:
            time_seq = T0300[num_seq_0300:num_seq_0300 + DAY]
            print(time_seq)
            h5_path = f"{nc_folder}/{time_seq[0]}/{province}/debug.h5"
            if not os.path.exists(h5_path):
                continue
            nc_paths = [f"{nc_folder}/{time_}/{province}/grid_hw08_{province}_2km_{time_}.nc" for time_ in time_seq]
            firemask, cloudmask, watermask = getMask(h5_path)
            if np.sum(firemask) <= 0:
                continue
            dataset = TestDataSet(nc_paths, 21, keywords)
            if len(dataset) <= 0:
                continue
            loader = DataLoader(dataset, batch_size=BS, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
            pred = []
            for batch in loader:
                patchs = batch["patch"]
                patchs = patchs.to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    labels_pred = net(patchs)
                     sm = torch.nn.Softmax()
                    labels_pred = sm(labels_pred)
                    pred_classes = torch.where(labels_pred > Fire_P, 1, 0)
                    pred_classes = torch.max(pred_classes, dim=1)[1]
                    pred_classes = pred_classes.cpu().numpy()
                    for pred_class in pred_classes:
                        pred.append(pred_class)
            pred = np.asarray(pred)
            pred = np.reshape(pred, (firemask.shape[0], firemask.shape[1]))
            pred = np.where((pred == 1), 1, 0)
            # saveRes(pred, f"D:/Fire-detection-result/Fireformer/{province}_{time_seq[0]}.mat")
            sub = firemask - pred
            add = firemask + pred
            TP = np.sum(np.where(add == 2, 1, 0))
            TN = np.sum(np.where(add == 0, 1, 0))
            FP = np.sum(np.where(sub == -1, 1, 0))
            FN = np.sum(np.where(sub == 1, 1, 0))
            acc = (TP + TN) / (TP + FP + TN + FN)
            recall = TP / (TP + FN)
            P = TP / (TP + FP)
            TP_ += TP
            TN_ += TN
            FP_ += FP
            FN_ += FN

            print("TP, TN, FP, FN", TP, TN, FP, FN)
            print("Acc:", acc)
            print("Recall", recall)
            print("P", P)

    print("The Sum:")
    print(TP_, "|", TN_, "|", FP_, "|", FN_, "|", img_num)
    acc = (TP_ + TN_) / (TP_ + FP_ + TN_ + FN_)
    recall = TP_ / (TP_ + FN_)
    P = TP_ / (TP_ + FP_)
    # F1 = (2*P*recall) / (P+recall)
    print("Acc:", acc)
    print("Recall", recall)
    print("P", P)
    print("Cost time:", all_cost_time)
    print("Images:", img_num)
    print("Net:", net.name)

