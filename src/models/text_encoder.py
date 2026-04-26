import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class MedicalTextEncoder(nn.Module):
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Bật Gradient Checkpointing để tiết kiệm VRAM, hỗ trợ Batch Size khổng lồ
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            
        self.feature_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Sử dụng embedding của token [CLS] để đại diện cho toàn bộ câu
        cls_output = outputs.last_hidden_state[:, 0, :]
        return cls_output
