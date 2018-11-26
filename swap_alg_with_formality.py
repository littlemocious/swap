import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler
import torch.functional as F
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
import pickle

EMBEDDING_DIM = 50
EPOCH_NUM = 20
BATCH_SIZE = 128
FORMALITY_SIZE = 10
#WEIGHT_DECAY = 0

#formalities: style_idx : style_formality_idx
#styles: style_name
#style_to_idx: style_name: style_idx
#pair_list: (style_name1, style_name2)
#pair_idx_list: (style_idx1, style_idx2)

swap_data = pickle.load(open("swap_data", "rb"))
formality = pickle.load(open("formality", "rb"))

styles = list(set(list(swap_data['origin_style']) 
                       + list(swap_data['replace_style'])))

style_to_idx = {style: i for i, style in enumerate(styles)}

pair_list = list(map(tuple,swap_data.values))

formality['style_index'] = formality.apply(lambda x: style_to_idx.get(x['style'], -1), axis = 1)
formality = formality[formality['style_index'] > -1]
formalities = formality.set_index('style_index')['formality'].to_dict()

STYLE_SIZE = len(styles)

pair_idx_list = [(style_to_idx[i[0]], style_to_idx[i[1]]) for i in pair_list] + \
                [(style_to_idx[i[1]], style_to_idx[i[0]]) for i in pair_list]
    
with open('styles', 'wb') as f:
    pickle.dump(styles, f)
    
with open('styleToIdx', 'wb') as f:
    pickle.dump(style_to_idx, f)


def get_input_layer_batch(word_idx_batch):
    x = torch.zeros([STYLE_SIZE, len(word_idx_batch)]).float()
    for i in range(len(word_idx_batch)):
        x[word_idx_batch[i], i] = 1.0
    return x

def get_input_form_batch(word_idx_batch):
    y = torch.zeros([FORMALITY_SIZE, len(word_idx_batch)]).float()
    for i in range(len(word_idx_batch)):
        y[formalities[word_idx_batch[i]] - 1, i] = 1.0
    return y

weight_decays = [0] + [10 ** i for i in range(-6,0)] + [5 * (10 ** i) for i in range(-6,-1)]

for j in range(len(weight_decays)):
    losses = []
    W1 = Variable(torch.randn([EMBEDDING_DIM, STYLE_SIZE]).float(), requires_grad=True)
    W2 = Variable(torch.randn([EMBEDDING_DIM, FORMALITY_SIZE]).float(), requires_grad=True)
    optimizer = torch.optim.Adam([W1, W2], lr = 1e-3, weight_decay = weight_decays[j])
    print('weight is' + str(weight_decays[j]))
    loss_function = torch.nn.NLLLoss()
    form_mat = get_input_form_batch(list(range(STYLE_SIZE)))

    for epoch in range(EPOCH_NUM):
        print('-----------------------' + str(epoch) + '-----------------------')
        data_loader = DataLoader(pair_idx_list, batch_size = BATCH_SIZE, shuffle = True)
        total_loss = 0
        i = 0
        for item1s, item2s in data_loader:
            item1s_idxs = get_input_layer_batch(item1s)
            item1s_form = get_input_form_batch(item1s.numpy())
            item2s_idxs = torch.tensor(item2s, dtype = torch.long)
            optimizer.zero_grad()
            z1 = torch.matmul(W1, item1s_idxs) + torch.matmul(W2, item1s_form)
            z2 = W1 + torch.matmul(W2, form_mat)
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
    with open('loss' + str(j), 'wb') as f:
    	pickle.dump(losses, f)