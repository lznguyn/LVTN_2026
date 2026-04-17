import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MedicalImageTextDataset(Dataset):
    def __init__(self, data_frame, image_transform=None, text_tokenizer=None, max_length=128, soft_labels=None):
        """
        data_frame: Pandas DataFrame chứa các cột ['image_path', 'report', 'cluster_id']
        soft_labels: Numpy array (N, K) chứa xác suất cụm mềm
        """
        self.df = data_frame
        self.image_transform = image_transform
        self.tokenizer = text_tokenizer
        self.max_length = max_length
        self.soft_labels = soft_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Đọc dữ liệu ảnh mới (Current)
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        
        # Tokenize (Số hóa) dạng văn bản báo cáo theo độ dài Max cho phép
        text = str(row['report'])
        text_inputs = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        # Lấy nhãn thông tin cụm bệnh cho Task Negative Sampling (nếu có)
        if 'cluster_id' in self.df.columns:
            cluster_id = torch.tensor(row['cluster_id'], dtype=torch.long)
        else:
            cluster_id = torch.tensor(-1, dtype=torch.long)
            
        # --- MỚI: Lấy nhãn cụm MỀM (Soft Label) ---
        if self.soft_labels is not None:
            soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)
        else:
            # Nếu không có nhãn mềm, tạo vector zero (không ảnh hưởng đến loss mới)
            soft_label = torch.zeros(50, dtype=torch.float32)
        
        return {
            'image': image,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'cluster_id': cluster_id,
            'soft_label': soft_label,
            'raw_report': text
        }
