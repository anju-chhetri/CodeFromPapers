import torch
import torch.nn as nn
from utils.readData import *
from utils.ResNet import ResNet18
import matplotlib.pyplot as plt
import numpy as np
from ood_metrics import auroc, fpr_at_95_tpr
import torchvision
import os

from metric import batch_jaccard_similarity
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 100
k = 64
number_of_layers = 4

TAU_PLUS = 0
TAU_MINUS = 1
LAMBDA_1 = 1
LAMBDA_2 = 1
GAMMA = np.arange(0.1, 0.9, 0.1).tolist()
# paths
pic_path = "/path/to/dataset"
image_path = "/path/to/save/images"

medical = True
iid_model="kvsair"
num_cmax_layers = 3

if medical:
    ood_dataset=[iid_model]

else:
    ood_dataset = ["cifar100"]   #define the ood dataset

model_name = "resnet"
 

def get_feature_map(name):
    
    def hook(model, input, output):
        feature_maps[name] = output.detach()
    return hook


def mag_freq_score(features_max, features_min, batch_size):
    RM_minus, RF_plus = 0, 0
    RM_plus = []
    RF_plus = []
    for features in features_max:

        max_values_per_channels = features.view(batch_size, k, -1).max(dim=2).values - TAU_PLUS
        masked_values = torch.where(max_values_per_channels>0, max_values_per_channels, torch.tensor(0.0))
        RM_plus.append(torch.mean(masked_values, dim=1))
        RF_plus.append(torch.mean(torch.where(max_values_per_channels>0, torch.tensor(1.0), torch.tensor(0.0)), dim=1))


    # min_values_per_channels = TAU_MINUS - features_min.view(batch_size, k, -1).max(dim=2).values
    min_values_per_channels = TAU_MINUS - features_min.view(batch_size, k, -1).min(dim=2).values - TAU_MINUS
    masked_values = torch.where(min_values_per_channels>0, min_values_per_channels, torch.tensor(0.0))
    RM_minus = torch.mean(masked_values, dim=1)
    RF_minus = torch.mean(torch.where(min_values_per_channels>0, torch.tensor(1.0), torch.tensor(0.0)), dim=1)
    return OOD_score_per_layer(RM_plus, RM_minus, RF_plus, RF_minus)


def OOD_score_per_layer(rm_plus, rm_minus, rf_plus, rf_minus):

    # return (rm_plus**LAMBDA_1) * (rm_minus**LAMBDA_1) * (rf_plus**LAMBDA_2) * (rf_minus**LAMBDA_2)
    return sum(rm_plus)**LAMBDA_1

def max_min_indices(weight, top_k_max_ind_list, top_k_min_ind, batch_size):

    max_weight = weight.view(weight.shape[0], weight.shape[1], -1).max(dim=2).values #out_channel, in_channel, 1(max)
    min_weight = weight.view(weight.shape[0], weight.shape[1], -1).min(dim=2).values #out_channel, in_channel, 1(max)
    per_img_weight_max = max_weight.repeat(batch_size, 1, 1) #batch, out, in
    per_img_weight_min = min_weight.repeat(batch_size, 1, 1)
    #See per_img_weight_max
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)

    # selected_max_list = []
    top_k_max_ind_layer_minus = []
    for top_k_max_ind in top_k_max_ind_list:
        selected_max = per_img_weight_max[batch_indices, top_k_max_ind] #batch, k, in
        # selected_max_list.append(selected_max)

        _, top_k_max_ind_val = torch.topk(selected_max.sum(dim=1), k) # batch*k
        top_k_max_ind_layer_minus.append(top_k_max_ind_val)

    selected_min = per_img_weight_min[batch_indices, top_k_min_ind]
    _, top_k_min_ind = torch.topk(selected_min.sum(dim=1), k, largest=False)

    return top_k_max_ind_layer_minus, top_k_min_ind


def get_layer_weights(model, layer_num: int, block_num: int, conv_num: int) -> torch.Tensor:

    layer = getattr(model, f'layer{layer_num}')
    block = layer[block_num]
    conv = getattr(block, f'conv{conv_num}')
    return conv.weight.data


def signal_strength_plot(convolutional_response, mode="r"):
    global image_name

    
    if not os.path.exists(os.path.join(image_path, image_name)):
        os.makedirs(os.path.join(image_path, image_name))
    modes = ["max", "mean", "min"]
    plt.figure(figsize=(10, 6))

    for m in modes:
        if m=="max":
            for i, la in enumerate(layers):
                max_values = torch.mean(convolutional_response[0][i].view(convolutional_response[0][i].shape[0], convolutional_response[0][i].shape[1], -1).max(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
                max_std = torch.std(convolutional_response[0][i].view(convolutional_response[0][i].shape[0], convolutional_response[0][i].shape[1], -1).max(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
                # plt.errorbar(range(max_values.shape[0]),max_values, max_std, label=f"iid_{iid_dataset[0]}", ls="-.")
                plt.plot(max_values, label=f"iid_{iid_dataset[0]}")
                for j in range(1, len(ood_dataset)+1):
                    max_values = torch.mean(convolutional_response[j][i].view(convolutional_response[j][i].shape[0], convolutional_response[j][i].shape[1], -1).max(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
                    max_std = torch.std(convolutional_response[j][i].view(convolutional_response[j][i].shape[0], convolutional_response[j][i].shape[1], -1).max(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
                    # plt.errorbar(range(max_values.shape[0]), max_values, max_std, label=f"ood_"+ood_dataset[j-1], ls="-.")
                    plt.plot(max_values, label=f"ood_"+ood_dataset[j-1])
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.title(f"Comparison of {m} values of iid and ood layer_{i+1}")
                plt.legend()
                plt.savefig(os.path.join(image_path, image_name, f"{m}_values_{mode}_{la}.png"), format="png", dpi=300)
                plt.clf()
        elif m=="min":
            for i, la in enumerate(layers):
                min_values = torch.mean(convolutional_response[0][i].view(convolutional_response[0][i].shape[0], convolutional_response[0][i].shape[1], -1).min(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
                std_min = torch.std(convolutional_response[0][i].view(convolutional_response[0][i].shape[0], convolutional_response[0][i].shape[1], -1).min(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W   
                # plt.errorbar(range(min_values), min_values, std_min, label=f"iid_{iid_dataset[0]}", ls="-.")
                plt.plot(min_values, label=f"iid_{iid_dataset[0]}")
                for j in range(1, len(ood_dataset)+1):
                    min_values = torch.mean(convolutional_response[j][i].view(convolutional_response[j][i].shape[0], convolutional_response[j][i].shape[1], -1).min(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
                    std_min = torch.std(convolutional_response[j][i].view(convolutional_response[j][i].shape[0], convolutional_response[j][i].shape[1], -1).min(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
                    # plt.errorbar(range(min_values), min_values, label=f"ood_"+ood_dataset[j-1],  ls="-.")
                    plt.plot(min_values, label=f"ood_"+ood_dataset[j-1])
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.title(f"Comparison of {m} values of iid and ood layer_{i+1}")
                plt.legend()
                plt.savefig(os.path.join(image_path, image_name, f"{m}_values_{mode}_{la}.png"), format="png", dpi=300)
                plt.clf()

        elif m=="mean":
            for i, la in enumerate(layers):
                mean_values = torch.mean(torch.mean(convolutional_response[0][i].view(convolutional_response[0][i].shape[0], convolutional_response[0][i].shape[1], -1), dim=2), dim=0).numpy() #across channel batch_size*channel_size*H*W
                std_mean = torch.std(torch.mean(convolutional_response[0][i].view(convolutional_response[0][i].shape[0], convolutional_response[0][i].shape[1], -1), dim=2), dim=0).numpy() #across channel batch_size*channel_size*H*W
                # plt.errorbar(len(mean_values), mean_values, std_mean, label=f"iid_{iid_dataset[0]}", ls="-.")
                plt.plot(mean_values, label=f"iid_{iid_dataset[0]}")
                for j in range(1, len(ood_dataset)+1):
                    mean_values = torch.mean(torch.mean(convolutional_response[j][i].view(convolutional_response[j][i].shape[0], convolutional_response[j][i].shape[1], -1), dim=2), dim=0).numpy() #across channel batch_size*channel_size*H*W
                    std_mean = torch.std(torch.mean(convolutional_response[j][i].view(convolutional_response[j][i].shape[0], convolutional_response[j][i].shape[1], -1), dim=2), dim=0).numpy() #across channel batch_size*channel_size*H*W
                    # plt.errorbar(len(mean_values), mean_values, std_mean, label=f"ood_"+ood_dataset[j-1], ls="-.")
                    plt.plot(mean_values, label=f"ood_"+ood_dataset[j-1])
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.title(f"Comparison of {m} values of iid and ood {la}")
                plt.legend()
                plt.savefig(os.path.join(image_path, image_name, f"{m}_values_{mode}_{la}.png"), format="png", dpi=300)
                plt.clf()


def plot_min_values(convolutional_response, mode="r"):
    global image_name

    
    if not os.path.exists(os.path.join(image_path, image_name)):
        os.makedirs(os.path.join(image_path, image_name))
    plt.figure(figsize=(10, 6))
    for i, la in enumerate(layers):
        min_values = torch.mean(convolutional_response[0][i].view(convolutional_response[0][i].shape[0], convolutional_response[0][i].shape[1], -1).min(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
        std_min = torch.std(convolutional_response[0][i].view(convolutional_response[0][i].shape[0], convolutional_response[0][i].shape[1], -1).min(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W   
        # plt.errorbar(range(min_values), min_values, std_min, label=f"iid_{iid_dataset[0]}", ls="-.")
        plt.plot(min_values, label=f"iid_{iid_dataset[0]}")
        for j in range(1, len(ood_dataset)+1):
            min_values = torch.mean(convolutional_response[j][i].view(convolutional_response[j][i].shape[0], convolutional_response[j][i].shape[1], -1).min(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
            std_min = torch.std(convolutional_response[j][i].view(convolutional_response[j][i].shape[0], convolutional_response[j][i].shape[1], -1).min(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
            # plt.errorbar(range(min_values), min_values, label=f"ood_"+ood_dataset[j-1],  ls="-.")
            plt.plot(min_values, label=f"ood_"+ood_dataset[j-1])
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(f"Comparison of min values of iid and ood {la}")
        plt.legend()
        plt.savefig(os.path.join(image_path, image_name, f"min_activated_min_values_{mode}_{la}.png"), format="png", dpi=300)
        plt.clf()

def plot_max_values(convolutional_response, mode="r"):
    global image_name

    if not os.path.exists(os.path.join(image_path, image_name)):
        os.makedirs(os.path.join(image_path, image_name))
    plt.figure(figsize=(10, 6))

    for i, la in enumerate(layers):
        for c_max_indices in range(num_cmax_layers):
            max_values = torch.mean(convolutional_response[0][i][c_max_indices].view(convolutional_response[0][i][c_max_indices].shape[0], convolutional_response[0][i][c_max_indices].shape[1], -1).max(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
            max_std = torch.std(convolutional_response[0][i][c_max_indices].view(convolutional_response[0][i][c_max_indices].shape[0], convolutional_response[0][i][c_max_indices].shape[1], -1).max(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
            # plt.errorbar(range(max_values.shape[0]),max_values, max_std, label=f"iid_{iid_dataset[0]}", ls="-.")
            plt.plot(max_values, label=f"iid_{c_max_indices+1}_{iid_dataset[0]}")
            for j in range(1, len(ood_dataset)+1):
                max_values = torch.mean(convolutional_response[j][i][c_max_indices].view(convolutional_response[j][i][c_max_indices].shape[0], convolutional_response[j][i][c_max_indices].shape[1], -1).max(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
                max_std = torch.std(convolutional_response[j][i][c_max_indices].view(convolutional_response[j][i][c_max_indices].shape[0], convolutional_response[j][i][c_max_indices].shape[1], -1).max(dim=2).values, dim=0).numpy() #across channel batch_size*channel_size*H*W
                # plt.errorbar(range(max_values.shape[0]), max_values, max_std, label=f"ood_"+ood_dataset[j-1], ls="-.")
                plt.plot(max_values, label=f"ood_{c_max_indices+1}"+ood_dataset[j-1])
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(f"Comparison of max values of iid and ood layer_{i+1}")
        plt.legend()
        plt.savefig(os.path.join(image_path, image_name, f"max_values_{mode}_{la}.png"), format="png", dpi=300)
        plt.clf()


def signal_plot(iid_loader, ood_loader):
    responses1 = [[], [], [], []]
    results = []
#For IID data
    count_val=0
    with torch.no_grad():
        for data, target in iid_loader:
            # if count_val<5:
            data = data.to(device)
            target = target.to(device)
            output = model(data).to(device)
            for i, j in enumerate(layers):
                responses1[i].append(feature_maps[j].detach().cpu())
            # penultimate[0].append(feature_maps["penultimate"])
            count_val+=1
        # if count_val==1:
        #     save_features(data, feature_maps, "iid")
            del output, data
        for i, _ in enumerate(responses1):
            responses1[i] = torch.cat(responses1[i])

        results.append(responses1)
        torch.cuda.empty_cache() 

        for ood in ood_loader:
            responses2 = [[], [], [], []]
            target_val = 0
            
            for data, target in ood:
                
                # if target_val<count_val:
                data = data.to(device)
                target = target.to(device)
                output = model(data).to(device)
                for i, j in enumerate(layers):
                    responses2[i].append(feature_maps[j].detach().cpu())

                target_val+=1
                # if target_va        l==1:
                #     save_features(data, feature_maps, "ood")
                del output, data
                torch.cuda.empty_cache()
            for i, _ in enumerate(responses1):
                responses2[i] = torch.cat(responses2[i])

            results.append(responses2)


        # signal_strength_plot(results, mode="all")
    #plot penultimate layer
    # breakpoint()
    # penultimate[0] = torch.cat(penultimate[0])
    # penultimate[1] = torch.cat(penultimate[1])
    # breakpoint()
    # penultimate[0] = torch.mean(penultimate[0], dim=).detach().cpu().numpy()
    # penultimate[1] = torch.mean(penultimate[1]).detach().cpu().numpy()
    # breakpoint()
    # plt.figure(figsize=(10,6))
    # plt.plot(penultimate[0], label="iid", color='b')
    # plt.plot(penultimate[1], label="ood", color='r')
    # plt.xlabel("node")
    # plt.ylabel("activation")
    # plt.legend()
    # plt.savefig(os.path.join(image_path, image_name, f"mean_values_plot_penultimate.png"), format="png", dpi=300)
    # plt.clf()

import os
from PIL import Image
def save_features(data, maps, name):
    main_path = f"/path/to/CORES/images"
    number_of_images = 5
    for images in range(number_of_images):
        image_path = os.path.join(main_path, str(images))
        if not os.path.exists(image_path):

            os.makedirs(image_path)

        original_image = data[images].detach().cpu().numpy()*255
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        original_image = (original_image*255).astype(np.uint8).transpose(2,1,0)

        org_img = Image.fromarray(original_image)
        org_img.save(os.path.join(image_path, f"image.png"))
        for la in layers:
            path = os.path.join(image_path, la)
            if not os.path.exists(path):
                os.makedirs(path)
            for num in range(maps[la][images].shape[0]):
                image_map = maps[la][images][num].detach().cpu().numpy()*255
                image_map = image_map.astype(np.uint8)
                img = Image.fromarray(image_map)
                img.save(os.path.join(path, f"filter_{num}.png"))
        


def score(loader):
    OOD_score = []
    filters = []
    feature_map_max = []
    feature_map_min = []
    max_features = []

    for la in range(number_of_layers):
        filters.append([])
        feature_map_min.append([])
        feature_map_max.append([])
        OOD_score.append([])

        for _ in range(num_cmax_layers):
            filters[la].append([])
            feature_map_max[la].append([])

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data).to(device)
        preds = softmax(output)
        
        c_max, c_min = preds.argmax(1), preds.argmin(1)


        # Number of max values; Get the indices of the sorted elements in descending order
        sorted_indices = torch.argsort(preds, dim=1, descending=True)
        c_max_list = sorted_indices[:, :num_cmax_layers]

        # # For penultimate layer
        weights = model.fc.weight.data

        top_k_max_ind_list = []
        for indices in range(num_cmax_layers):
            max_indices = c_max_list[:,indices]
            _, top_k_max_ind = torch.topk(weights[max_indices], k)
            top_k_max_ind_list.append(top_k_max_ind)
        _, top_k_min_ind = torch.topk(weights[c_min], k)

        for layer_num in range(number_of_layers,0, -1):
            total_response = feature_maps[f"layer{layer_num}"] # batch, channel, H, W
            batch_indices = torch.arange(total_response.shape[0]).unsqueeze(1).expand(-1, k)
            
            selected_features_max_list = []

            for con, top_k_max_ind in enumerate(top_k_max_ind_list):
                selected_features_max = total_response[batch_indices, top_k_max_ind]
                selected_features_max_list.append(selected_features_max)
                filters[layer_num-1][con].append(top_k_max_ind)
                feature_map_max[layer_num-1][con].append(selected_features_max)
            
            selected_features_min = total_response[batch_indices, top_k_min_ind] #Shape [batch, k, h, w]            
            feature_map_min[layer_num-1].append(selected_features_min)
            OOD_score[number_of_layers-layer_num].append(mag_freq_score(selected_features_max_list, selected_features_min, total_response.shape[0]))
            
            for block_num in range(1,-1, -1):
                for conv_num in range(2,0, -1):
                    weight = get_layer_weights(model, layer_num, block_num, conv_num)
                    top_k_max_ind_list, top_k_min_ind = max_min_indices(weight, top_k_max_ind_list, top_k_min_ind, total_response.shape[0])

        # OOD_score has reverse of layers  

    for i in range(number_of_layers):
        OOD_score[i] = torch.cat(OOD_score[i]).detach().cpu().numpy()
        feature_map_min[i] = torch.cat(feature_map_min[i]).detach().cpu().numpy()
        for c_max_layer in range(num_cmax_layers):
            filters[i][c_max_layer] = torch.cat(filters[i][c_max_layer]).detach().cpu().numpy()
            feature_map_max[i][c_max_layer] = torch.cat(feature_map_max[i][c_max_layer]).detach().cpu()
    
    return OOD_score[::-1], filters, feature_map_max, feature_map_min
 

def get_metric(iid_scores, ood_scores):
    fpr =[]
    auroc_val = []
    breakpoint()
    for G in GAMMA:
        iid_labels = [0]*len(iid_scores)
        ood_labels = [1]*len(ood_scores)

        labels = iid_labels+ood_labels
        scores = (1-(iid_scores>G).astype(int)).tolist()+(1-(ood_scores>G).astype(int)).tolist()
        fpr.append(fpr_at_95_tpr(scores, labels))
        auroc_val.append(auroc(scores, labels))
    return fpr, auroc_val

def plot_frequency_plot(filter: list): #filter = [iid_filter, ood_filter1, ..., ood_filter_n]
    global image_name
    for names in ood_dataset:
        image_name=image_name+'_'+names
    from collections import Counter

    if not os.path.exists(os.path.join(image_path, image_name)):
        os.makedirs(os.path.join(image_path, image_name))

    for i in range(len(filter[0])): # number of layers plot is done across layers
        for j, f in enumerate(filter): # plot each dataset for each layer
            if j==0:
                label=f"iid_"+iid_model

            else:
                label= f"ood_"+ood_dataset[j-1]

            for c_max_len in range(num_cmax_layers):
                flatten_indices = f[i][c_max_len].flatten()
                filter_count = Counter(flatten_indices)
                label = label+f"_{c_max_len+1}"
                plt.bar(filter_count.keys(),filter_count.values(), alpha=0.5, label=label)
        plt.xlabel("Filter index")
        plt.ylabel("Frequency")
        plt.title(f"Frequency of filters layer_{i+1}")
        plt.legend()
        plt.savefig(os.path.join(image_path, image_name, f"frequency_plot_layer_{i+1}.png"), format="png", dpi=300)  
        plt.clf()

def load_dataset(name):
    n_class=10
    if name == "cifar10":
        _ , _, loader = read_dataset(batch_size=batch_size, pic_path=pic_path)
    
    elif name == "lsun":
        print("LSUN")
        n_class=9
        _, loader = read_lsun_data(os.path.join(pic_path, "lsun"), batch_size=batch_size)

    elif name == "mnist":
        print("MNIST")
        _ , _, loader = read_mnist_dataset(batch_size=batch_size, pic_path=pic_path)
    
    elif name == "svhn":
        print("here Svhn")
        _, loader = read_ood_data(batch_size=batch_size, pic_path=pic_path)

    elif name=="cifar100":
        print("CIFAR100")
        _,_,loader = read_cifar100_dataset(pic_path, batch_size=batch_size)
        n_class=100
    
    elif name=="mmnist":
        print("MNIST")
        _, _, loader = read_mnist_dataset(pic_path, batch_size=batch_size)

    elif name=="fashionmnist":
        print("Fashion MNIST")
        _, _, loader = read_fashion_mnist_dataset(pic_path, batch_size=batch_size)

    elif name=="imagenet":
        print("Imagenet")
        n_class = 1000
        loader = read_imagenet_dataset(pic_path, batch_size=batch_size)

    elif name=="places365":
        print("Places365")
        n_class = 365
        _ , loader = read_places365_dataset(pic_path, batch_size=batch_size)
    
    elif name=="kvsair":
        print("Kvsair")
        n_class = 3
        _, id_loader, ood_loader = read_kvsair_dataset(pic_path, batch_size=batch_size)
        loader = (id_loader, ood_loader)

    elif name=="gastrovision":
        print("Gastrovision")
        n_class = 11
        _, id_loader, ood_loader = read_gastrovision_dataset(pic_path, batch_size=batch_size)
        loader = (id_loader, ood_loader)
    return loader, n_class


if __name__=="__main__":
    #define model
    #Dataset stuff
    checkpoint_dir = f"/path/to/checkpoint"
    iid_dataset = [iid_model]
    image_name = f"{model_name}_{iid_model}"
    
    if not medical:
        for d_set in iid_dataset:
            iid_loader, n_class = load_dataset(d_set)
        
        n_class = n_class
        ood_datasets = []
        for o_set in ood_dataset:
            loader, _ = load_dataset(o_set)
            ood_datasets.append(loader)
    
    else:
        loader, n_class = load_dataset(ood_dataset[0])
        iid_loader = loader[0]
        ood_datasets = [loader[1]]

    if model_name=="resnet":
        if medical:
            model = torchvision.models.resnet18()
        # model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=1, bias=False)
        else:
            model = ResNet18() 
            model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

        model.fc = torch.nn.Linear(512, n_class)

    #define
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir,"best_model.pt")))
    
    elif model_name=="vgg":
        model = torchvision.models.vgg16(weights='IMAGENET1K_V1')

    elif model_name=="resnet50":
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    
    elif model_name=="resnet34":
        model = torchvision.models.resnet34(weights="IMAGENET1K_V1")

    model = model.to(device)
    softmax = nn.Softmax(dim=1)
    model.eval()
    #Register hooks:
    if model_name=="resnet":

        layers = ["layer1", "layer2", "layer3",  "layer4"]

        model.layer1[1].conv2.register_forward_hook(get_feature_map(layers[0]))
        model.layer2[1].conv2.register_forward_hook(get_feature_map(layers[1]))
        model.layer3[1].conv2.register_forward_hook(get_feature_map(layers[2]))
        model.layer4[1].conv2.register_forward_hook(get_feature_map(layers[3]))
    # model.fc.register_forward_hook(get_feature_map("penultimate"))
    
    if model_name == "vgg":
        layers = ["layer_7", "layer_14", "layer_21",  "layer28"]

        model.features[7].register_forward_hook(get_feature_map(layers[0]))
        model.features[14].register_forward_hook(get_feature_map(layers[1]))
        model.features[21].register_forward_hook(get_feature_map(layers[2]))
        model.features[28].register_forward_hook(get_feature_map(layers[3]))

    if model_name == "resnet50":
        layers = ["layer1", "layer2", "layer3",  "layer4"]

        model.layer1[2].conv3.register_forward_hook(get_feature_map(layers[0]))
        model.layer2[3].conv2.register_forward_hook(get_feature_map(layers[1]))
        model.layer3[5].conv3.register_forward_hook(get_feature_map(layers[2]))
        model.layer4[2].conv3.register_forward_hook(get_feature_map(layers[3]))

    if model_name == "resnet34":
        layers = ["layer1", "layer2", "layer3",  "layer4"]

        model.layer1[2].conv2.register_forward_hook(get_feature_map(layers[0]))
        model.layer2[3].conv2.register_forward_hook(get_feature_map(layers[1]))
        model.layer3[5].conv2.register_forward_hook(get_feature_map(layers[2]))
        model.layer4[2].conv2.register_forward_hook(get_feature_map(layers[3]))

    feature_maps = {}


    iid_ood_score, iid_filter, iid_feature_map, iid_min_feature_map= score(iid_loader)

    all_filters= []
    feature_maps_max=[]
    feature_maps_min = []
    all_filters.append(iid_filter)
    feature_maps_max.append(iid_feature_map)
    feature_maps_min.append(iid_min_feature_map)

    for ood in ood_datasets:
        ood_ood_score, ood_filter, fm, ood_min_feature_map = score(ood)
        all_filters.append(ood_filter)   #ood filter = (num_of_dataset * len(layers) * number of C_max*samples*number of relevant filters)
        feature_maps_max.append(fm)
        feature_maps_min.append(ood_min_feature_map)

    plot_frequency_plot(all_filters)
    
    # print(f"mean of all layers: iid {np.mean(iid_ood_score, axis=1)} ood: {np.mean(ood_ood_score, axis=1)}")
    # print(f"Mean across all the data samples: iid {np.mean(np.mean(iid_ood_score, axis=1))}, ood: {np.mean(np.mean(ood_ood_score, axis=1))}")
    iid_ood_score = batch_jaccard_similarity(all_filters[0][-1][0], all_filters[0][-1][1])
    ood_ood_score = batch_jaccard_similarity(all_filters[1][-1][0], all_filters[1][-1][1])
    # fpr, roc = get_metric(iid_ood_score[-1], ood_ood_score[-1])
    fpr, roc = get_metric(iid_ood_score, ood_ood_score)

    for i in range(len(GAMMA)):

        print(f"GAMMA: {GAMMA[i]} fpr: {fpr[i]} ROC:{roc[i]}")

    
    # plot_max_values(feature_maps_max, mode=f"r_2max_{k}")
    # signal_plot(iid_loader, ood_datasets)

    # plot_min_values(feature_maps_min, mode=f"r_2max_{k}")

