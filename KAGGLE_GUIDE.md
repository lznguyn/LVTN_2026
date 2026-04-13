# Hướng dẫn chạy trên Kaggle

Dự án này sử dụng các mô hình lớn (SwinV2, ClinicalBERT) nên yêu cầu cấu hình phần cứng mạnh. Kaggle cung cấp GPU miễn phí, là một lựa chọn lý tưởng để huấn luyện.

## 1. Cấu hình Notebook
1. Tạo một Kaggle Notebook mới.
2. Trong phần **Session options** (hoặc Settings):
   - **Accelerator**: Chọn **GPU P100** hoặc **GPU T4x2**. (Bắt buộc)
   - **Internet**: Bật **Internet on** (Cần thiết để tải dữ liệu, pre-trained weights và thư viện).

## 2. Chuẩn bị mã nguồn
Bạn có thể mang mã nguồn dự án vào Kaggle bằng cách clone trực tiếp từ GitHub:
```bash
!git clone https://github.com/your-username/LVTN_2026.git
%cd LVTN_2026
```
*(Nếu không dùng Git, nén thư mục dự án thành file .zip, đưa lên dạng Add Data và copy vào `/kaggle/working/LVTN_2026`)*

## 3. Về vấn đề Dữ liệu (Dataset)
Vì trong file `scripts/prepare_dataset.py` hệ thống đã có code **tự động tải** trực tiếp dữ liệu Chest X-Ray từ trang chủ của OpenI NIH, bạn có 2 lựa chọn:

### CÁCH 1: Để code tự động tải (Tự động nhưng tốn dung lượng)
- **Ưu điểm**: Nhanh, gọn, thực hiện 100% bằng code. Chỉ cần chạy file `prepare_dataset.py`, nó sẽ tải và giải nén thẳng vào thư mục `data/raw/` và sinh ra `data/processed/` bên trong `/kaggle/working/LVTN_2026/`.
- **Nhược điểm**: 
  - Dung lượng Output của Kaggle (`/kaggle/working`) chỉ có **tối đa 20GB**. Việc tải / nén / giải nén vài GB hình ảnh trực tiếp ở đây cộng thêm các checkpoint mô hình sinh ra có thể khiến Notebook bị sập vì đầy ổ cứng.
  - Khi tắt Kaggle (kết thúc session), mọi thứ có thể bị xóa đi khiến bạn phải load lại từ đầu vào lần sau.
- **Cấu hình**: Bạn **không cần** sửa file `configs/default.yaml` vì nó đã cấu hình sẵn thư mục tương đối `data/...`.

### CÁCH 2: Dùng chức năng Dataset chuyên dụng của Kaggle (Khuyên dùng)
Nếu muốn chuyên nghiệp và an toàn hơn:
1. Bạn chạy file `prepare_dataset.py` một lần trên máy tính ở nhà. Lấy bộ thư mục ảnh và file `.csv` kết quả nén lại thành 1 file zip.
2. Tại Kaggle, nhấn **Add Data -> Upload a Dataset** rồi tải file zip đó lên.
3. **Ưu điểm lớn**: Dữ liệu sẽ thuộc thư mục **`/kaggle/input/`**, phân vùng này Kaggle cho nhiều dung lượng hơn và đặc biệt là **không tính** vào giới hạn 20GB của `/kaggle/working/`. Bạn tha hồ lưu Checkpoint mô hình! 
4. **Cấu hình**: Cần mở `configs/default.yaml` để sửa thư mục data trỏ về `/kaggle/input/...`

## 4. Huấn luyện
Tại các cell tiếp theo, chạy lần lượt:

```bash
# Cài đặt thư viện
!pip install -r requirements.txt

# Tiền xử lý dữ liệu và Phân cụm (Clustering)
# Lưu ý: Nếu dùng data tải sẵn (Cách 2), nhớ khóa 2 dòng tải trên mạng trong file python lại.
!python scripts/prepare_dataset.py
!python scripts/create_clusters.py

# Khởi chạy Huấn luyện Stage 1
!python scripts/train.py

# Đánh giá sau huấn luyện
!python scripts/evaluate.py
```
> 💡 **Lưu ý thời gian**: Kaggle giới hạn max 12 tiếng mỗi Session. Đảm bảo config lưu checkpoint từng Epoch để không bị mất kết quả!
