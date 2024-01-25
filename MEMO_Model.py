from transformers import GPT2PreTrainedModel, GPT2Model
import torch.nn as nn
from torch.nn import CrossEntropyLoss,MSELoss
from dataset import *

class MEMO_Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super(MEMO_Model, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.img_ff = nn.Linear(512, config.n_embd) 
        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
 
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

    def forward(self, bz, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None,input_img=None, tokenizer=None, input_construct=None,emo_labels=None):
        history_img_embs = self.img_ff(input_img)
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               input_img=history_img_embs,
                                               input_construct=input_construct,
                                               tokenizer=tokenizer)
        txt_hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(txt_hidden_states)
        outputs = (lm_logits,)
    
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),shift_labels.view(-1))
       
            
                
            if emo_labels is not None:
               
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels_emo = emo_labels[..., 1:].contiguous()
                loss_fct_emo = CrossEntropyLoss(ignore_index=-1)
                loss_emo = loss_fct_emo(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels_emo.view(-1))    
        
            outputs = (loss,) + outputs
        
        return outputs 
    