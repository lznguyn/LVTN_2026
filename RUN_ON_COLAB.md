# 🚀 HƯỚNG DẪN CHẠY TOÀN BỘ LUẬN VĂN TRÊN GOOGLE COLAB 🚀
*Dành riêng cho dữ liệu MIMIC-CXR chia sẻ qua Google Drive*

---

## BƯỚC 1: CHUẨN BỊ MÔI TRƯỜNG TRÊN GOOGLE DRIVE (Làm 1 lần)
1. Truy cập vào tài khoản Google Drive của bạn.
2. Tìm bộ dữ liệu nằm ở tab **"Được chia sẻ với tôi" (Shared with me) Nhật ký**. Tên thư mục là: `db_MIMIC_CXRCXR`.
3. Nhấp chuột phải vào thư mục đó $\rightarrow$ Chọn **"Tổ chức" (Organize)** $\rightarrow$ Chọn **"Thêm lối tắt" (Add Shortcut)**.
4. Chọn vị trí lưu là **"Drive của tôi" (My Drive)** $\rightarrow$ Bấm Thêm (Lưu ý: Không đổi tên Lối tắt).

---

## BƯỚC 2: TẢI CODE LÊN GOOGLE DRIVE
1. Trên máy tính, bảo đảm bạn đã chạy lệnh cài đặt dọn dẹp Git lúc nãy.
2. Vẫn trên thư mục mẹ ở máy tính, **Nén toàn bộ thư mục dự án (chứa mục `scripts/`, `src/`...) thành file đuôi `.zip`**. Đặt tên nén là: `Multimodal_Negative_Sampling.zip`.
3. Tải cục file ZIP nhỏ bé này lên ổ gốc **Drive của tôi (My Drive)** trên Google Drive. 
*(Vậy là trong ổ gốc của mây ảo nhà bạn đang có 2 thứ: Thư mục lối tắt `db_MIMIC_CXRCXR` và cục code nén `Multimodal_Negative_Sampling.zip`)*.

---

## BƯỚC 3: MỞ GOOGLE COLAB VÀ THIẾT LẬP MÁY ẢO T4
1. Vào trang bìa [Colab](https://colab.research.google.com/). Tạo một cái **Trang tính (Notebook) mới**.
2. Góc trên màn hình, chọn **Thời gian chạy (Runtime)** $\rightarrow$ **Thay đổi Loại Thời gian chạy (Change runtime type)**.
3. Ở ô **Tiện ích phần cứng (Hardware accelerator)**, chọn **GPU T4** $\rightarrow$ Bấm *Lưu (Save)*.

---

## BƯỚC 4: 5 KHỐI CELL CHẠY TOÀN BỘ QUY TRÌNH
Trên giao diện Colab, bạn hãy mở 5 cái Dòng mã (Code Cell) riêng biệt và dán lần lượt 5 đoạn này vào, nhớ ấn nút **`Play ▶`** ở từng khối để chạy theo thứ tự nhé!

### 💻 Block 1: Cấp quyền truy cập Drive và Giải nén Code
```python
from google.colab import drive
drive.mount('/content/drive')

# Xóa rác Cũ (nếu có để chống lỗi kẹt đường dẫn)
!rm -rf /content/Multimodal
# Giải nén Code từ Drive vào siêu ổ cứng SSD của Colab để tốc độ tải x10 lần
!unzip -q "/content/drive/MyDrive/Multimodal_Negative_Sampling.zip" -d "/content/Multimodal"
```

### 💻 Block 2: Cài đặt Thư Viện AI
```bash
%cd /content/Multimodal
!pip install -r requirements.txt
```

### 💻 Block 3: Máy ủi Data MIMIC-CXR  🌟(Rất quan trọng)
*Quá trình này bóc tách file ZIP khổng lồ bên trong Thư viện Drive của bạn, và tự động ráp Ảnh gốc vào file Báo cáo *.txt để thành cục Excel chung cuối cùng.*
```bash
!python scripts/prepare_mimic_colab.py
```

### 💻 Block 4: Máy nhồi Nhóm Bệnh Học (Sức mạnh NLP)
*Đang sử dụng trí tuệ Sentence-BERT nhúng text và dập thuật toán K-Means chia 14 bệnh ở phổi để gán Mác Clustering Masking cho Mô hình học.*
```bash
!python scripts/create_clusters.py
```

### 💻 Block 5: Lễ Khai mạc Huấn Luyện Epochs 
*Card T4 sẽ rên gầm dữ dội hàng tiếng đồng hồ để chạy vòng lặp Contrastive Loss này.*
```bash
!python scripts/train.py
```

---

## BƯỚC CUỐI CÙNG: MÓC MODEL TRẢ VỀỔ NHÀ
Sau khoảng 4-8 tiếng (sau khi Block 5 chạy tới 100% hoàn thành), file Siêu Bộ Lọc `.pth` xịn xò nhất đã được đúc thành công. Nó vẫn đang nằm lang thang trên Server Google. 
Hãy dán Code dưới đây vào Cell số 6 để chép đè Cục Vàng đó về Drive ảo của bạn đem về nộp LVTN nhé!

```bash
!cp /content/Multimodal/checkpoints/best_model.pth /content/drive/MyDrive/best_model.pth
print("CHÚC MỪNG TÂN CỬ NHÂN! MODEL ĐÃ ĐƯỢC CẤP CỨU VỀ DRIVE AN TOÀN!")
```
