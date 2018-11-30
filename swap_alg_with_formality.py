from get_swap_data import fetch_data
from configparser import RawConfigParser
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.functional as F
import torch.nn.functional as F
from torch.autograd import Variable

EMBEDDING_DIM = 50
EPOCH_NUM = 20
BATCH_SIZE = 128
FORMALITY_SIZE = 10
WEIGHT_DECAY = 0
SHAPE_SIZE = 10

secrets = RawConfigParser()
secrets.read("secrets.ini")

def prepare_feature(data, feature_col, style_col = 'style'):
    data['style_index'] = data.apply(lambda x: style_to_idx.get(x[style_col], -1), axis = 1)
    data = data[data['style_index'] > -1]
    data = data.set_index('style_index')[feature_col].to_dict()
    return(data)

def get_input_layer_batch(word_idx_batch):
    x = torch.zeros([STYLE_SIZE, len(word_idx_batch)]).float()
    for i in range(len(word_idx_batch)):
        x[word_idx_batch[i], i] = 1.0
    return x

def get_input_feat_batch(word_idx_batch, feat_size, feat_list):
    y = torch.zeros([feat_size, len(word_idx_batch)]).float()
    for i in range(len(word_idx_batch)):
        y[feat_list[word_idx_batch[i]] - 1, i] = 1.0
    return y

formality, shape, swap_data = fetch_data(secrets)

swap_data.iloc[swap_data[swap_data['replace_style'] == 'CT34'].index, 1] = 'CT18'
swap_data.iloc[swap_data[swap_data['origin_style'] == 'CT34'].index, 0] = 'CT18'

styles = list(set(list(swap_data['origin_style']) + list(swap_data['replace_style'])))
STYLE_SIZE = len(styles)

style_to_idx = {style: i for i, style in enumerate(styles)}

pair_list = list(map(tuple,swap_data.values))
        
for i in pair_list:
    if i[0] == i[1]:
        pair_list.remove(i)

pair_idx_list = [(style_to_idx[i[0]], style_to_idx[i[1]]) for i in pair_list] + \
                [(style_to_idx[i[1]], style_to_idx[i[0]]) for i in pair_list]

shapes = prepare_feature(shape, 'shape')

shape_to_idx = {shape: i for i, shape in enumerate(set(shapes.values()))}

for idx in shapes:
    shapes[idx] = shape_to_idx[shapes[idx]]

formalities = prepare_feature(formality, 'formality')
weight_decays = [0, 1e-5, 5e-5, 1e-4]

for j in range(len(weight_decays)):
    losses = []
    W1 = Variable(torch.randn([EMBEDDING_DIM, STYLE_SIZE]).float(), requires_grad=True)
    W2 = Variable(torch.randn([EMBEDDING_DIM, FORMALITY_SIZE]).float(), requires_grad=True)
    W3 = Variable(torch.randn([EMBEDDING_DIM, SHAPE_SIZE]).float(), requires_grad=True)
    optimizer = torch.optim.Adam([W1, W2, W3], lr = 1e-3, weight_decay = weight_decays[j])
    print('weight is ' + str(weight_decays[j]))
    loss_function = torch.nn.NLLLoss()
    form_mat = get_input_feat_batch(list(range(STYLE_SIZE)), FORMALITY_SIZE, formalities)
    shape_mat = get_input_feat_batch(list(range(STYLE_SIZE)), SHAPE_SIZE, shapes)
    
    for epoch in range(EPOCH_NUM):
        print('-----------------------' + str(epoch) + '-----------------------')
        data_loader = DataLoader(pair_idx_list, batch_size = BATCH_SIZE, shuffle = True)
        total_loss = 0
        i = 0
        for item1s, item2s in data_loader:
            item1s_idxs = get_input_layer_batch(item1s)
            item1s_form = get_input_feat_batch(item1s.numpy(), FORMALITY_SIZE, formalities)
            item1s_shape = get_input_feat_batch(item1s.numpy(), SHAPE_SIZE, shapes)
            item2s_idxs = torch.tensor(item2s, dtype = torch.long)
            optimizer.zero_grad()
            z1 = torch.matmul(W1, item1s_idxs) + torch.matmul(W2, item1s_form) + torch.matmul(W3, item1s_shape)
            z2 = W1 + torch.matmul(W2, form_mat) + torch.matmul(W3, shape_mat)
            z3 = torch.matmul(z2.t(), z1)
            log_probs = torch.nn.functional.log_softmax(z3, dim=0).t()
            loss = loss_function(log_probs, item2s_idxs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            i += 1
            if not i % 10000:
                print(i)
        losses.append(total_loss)   
        print(f'Loss at epoch {epoch}: {total_loss}')
        torch.save(W1,'w1' + str(j) + '.pt')
        torch.save(W2,'w2' + str(j) + '.pt')
        torch.save(W3,'w3' + str(j) + '.pt')