import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class MedicalImageTextDataset(Dataset):
    def __init__(self, data_frame, image_transform=None, text_tokenizer=None, max_length=128):
        """
        data_frame: Pandas DataFrame chứa các cột ['image_path', 'report', 'cluster_id']
        """
        self.df = data_frame
        self.image_transform = image_transform
        self.tokenizer = text_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Đọc dữ liệu ảnh mới (Current)
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        
        # Đọc dữ liệu ảnh cũ (Old/Baseline) - Nếu không có thì lấy chính ảnh hiện tại
        old_img_path = row.get('old_image_path', img_path)
        # Kiểm tra nếu old_img_path là NaN hoặc rỗng
        if not isinstance(old_img_path, str) or old_img_path.strip() == "" or old_img_path.lower() == "nan":
            old_img_path = img_path
            
        try:
            old_image_pil = Image.open(old_img_path).convert('RGB')
        except Exception:
            # Fallback nếu đường dẫn ảnh cũ lỗi
            old_image_pil = Image.open(img_path).convert('RGB')
            
        if self.image_transform:
            old_image = self.image_transform(old_image_pil)
        else:
            old_image = transforms.ToTensor()(old_image_pil) # Safe fallback
            
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
        
        return {
            'image': image,
            'image_old': old_image,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'cluster_id': cluster_id,
            'raw_report': text
        }
