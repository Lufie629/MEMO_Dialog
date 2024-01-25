import json 
from transformers import BertTokenizer
import torch 
from torch.utils.data import Dataset 
import numpy as np 
import copy 
import os
import torch
import json
import numpy as np
from transformers import BertTokenizer,GPT2Config,AdamW
from utils import accuracy_compute, AverageMeter, meme_classify_accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = ['[IMG]', '[TAG]','[V0]','[V1]','[V2]','[V3]','[V4]', '[PAD]']
SPECIAL_TOKENS_DICT = {'additional_special_tokens':['[IMG]', '[TAG]','[V0]','[V1]','[V2]','[V3]','[V4]'],'pad_token': '[PAD]'}

def tokenize(obj, tokenizer):
    return tokenizer.encode(obj)
    

def get_data(tokenizer, data_path, meme_feature_path, emotion_dict):
    dialog_data = json.load(open(data_path, 'r', encoding='utf-8')) 
    print("get_data' type: ", type(dialog_data))
    dialog_list = [] 
    ids = 0
    
    search_info = 0
    
    for idx in dialog_data.keys():
        dialog = dialog_data[idx] 
        history = [] 
        for i in range(len(dialog)): 
            if 'txt' in dialog[i].keys(): 
           
                dialog[i]['txt'] = tokenize(dialog[i]['txt'], tokenizer) 
     
            if 'emotion_id' in dialog[i].keys():                
             
                
                dialog[i]['emotion'] = emotion_dict[str(int(dialog[i]['emotion_id']))]
               
                dialog[i]['emotion'] = tokenize(dialog[i]['emotion'], tokenizer)
            else:
                dialog[i]['emotion'] = tokenize("中性", tokenizer)
            if i == 0:
                history.append(dialog[i]) 
                continue 
     
            pair = {'history': copy.deepcopy(history), 'answer': copy.deepcopy(dialog[i])} 
            dialog_list.append(pair) 
            history.append(dialog[i])
            search_info += 1
         
   
    print("search_info:",search_info)
    id2feature = json.load(open(meme_feature_path, 'r', encoding='utf-8')) 
    return dialog_list, id2feature 


class MODDataset(Dataset): 
    def __init__(self, dialogs, id2feature, tokenizer,batch_first=True): 
        self.dialogs = dialogs 
        self.id2feature = id2feature 
        self.tokenizer = tokenizer 
        self.batch_first = batch_first
        self.pad = tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.dialogs) 
    
    def __getitem__(self, index): 
        his = copy.deepcopy(self.dialogs[index]['history']) 
        ans = copy.deepcopy(self.dialogs[index]['answer']) 
        instance = {}
        history_txt, token_type_ids, labels, history_img, emo_labels = build_input_from_segments(history=his, answer=ans, tokenizer=self.tokenizer, id2feature=self.id2feature) 
        instance["input_ids"] = history_txt
        instance["token_type_ids"] = token_type_ids
        instance["lm_labels"] = labels
        instance["history_img"] = history_img
        instance["emo_labels"] = emo_labels


        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        
        emo_labels = pad_sequence(
            [torch.tensor(instance["emo_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)

        history_img = pad_sequence(
            [torch.from_numpy(np.array(instance["history_img"])).float()  for instance in batch],
            batch_first=self.batch_first, padding_value=0.0)
        return input_ids, token_type_ids, labels, history_img, emo_labels


# build input type from data 
def build_input_from_segments(history, tokenizer, id2feature, answer=None): 
    img, tag,v0,v1,v2,v3,v4 = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1])

    bos, eos, speaker1, speaker2 = 0, 2, 13086, 13087
    history_txt = []
    history_img = [] 
    labels = []
    token_type_ids = [] 
    emo_labels = []
    

    ans_len = 4 
    if answer is not None and 'txt' in answer.keys():
        ans_len += len(answer['txt']) 
    
    for i in range(len(history)-1, -1, -1): 
        
        # evaluate the length 
        cur_len = 4 
        if 'txt' in history[i].keys():
            cur_len += len(history[i]['txt']) 
        if len(token_type_ids) + ans_len + cur_len > 500: 
            break 

        if history[i]['speaker_id'] == '[speaker1]': 
            speaker_id = speaker1 
        else:
            speaker_id = speaker2 
        if 'img_id' in history[i].keys(): 
            history_txt = [img] + history_txt
            history_img = [id2feature[history[i]['img_id']]] + history_img
            token_type_ids = [img] + token_type_ids 
            labels = [-1] + labels 
            emo_labels = [-1] + emo_labels

        if 'emotion_id' in history[i].keys(): 
            content = [v0]+[v1]+[v2]+[v3]+[v4]+history[i]['emotion']
            history_txt = content + history_txt 
            token_type_ids = [speaker_id] * len(content) + token_type_ids 
            labels = [-1] * len(content) + labels
            emo_labels = [-1] * len(content) + emo_labels
        if 'txt' in history[i].keys(): 
            content =history[i]['txt']
            history_txt = content + history_txt 
            token_type_ids = [speaker_id] * len(content) + token_type_ids 
            labels = [-1] * len(content) + labels 
            emo_labels = [-1] * len(content) + emo_labels
        else: 
            content = []
            history_txt = content + history_txt 
            token_type_ids = [speaker_id] * len(content) + token_type_ids 
            labels = [-1] * len(content) + labels 
            emo_labels = [-1] * len(content) + emo_labels
    
        history_txt = [speaker_id] + history_txt 
        token_type_ids = [speaker_id] + token_type_ids 
        labels = [-1] + labels 
        emo_labels = [-1] + emo_labels
    if answer is not None: 
        if answer['speaker_id'] == '[speaker1]': 
            speaker_id = speaker1 
        else:
            speaker_id = speaker2 
    
        history_txt += [speaker_id] 
        token_type_ids += [speaker_id] 
        labels +=[-1]
        emo_labels+=[-1]
        if 'txt' in answer.keys(): 
            content =answer['txt']
            history_txt += content 
            token_type_ids += [speaker_id] * len(content) 
            labels += content 
            emo_labels += [-1]*len(content)
            history_txt+=[eos]
            token_type_ids += [eos]
            labels += [eos] 
            emo_labels += [-1]
        else: 
            content =[] 
            history_txt += content 
            token_type_ids += [speaker_id] * len(content) 
            labels += content 
            emo_labels += [-1]*len(content)
        if 'emotion_id' in answer.keys(): 
            content = [v0]+[v1]+[v2]+[v3]+[v4]+answer['emotion']
            history_txt += content 
            token_type_ids += [speaker_id] * len(content) 
            labels += [-1]*len(content)
            emo_labels += content 


    
    
    history_txt = [bos] + history_txt
    token_type_ids = [bos] + token_type_ids
    labels = [-1] + labels
    emo_labels = [-1] + emo_labels
  

    if history_img==[]:
        history_img += [[0.0]*512]
 
    return history_txt, token_type_ids, labels, history_img, emo_labels