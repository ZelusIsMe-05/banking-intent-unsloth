# Project 2: FINE-TUNING INTENT DETECTION MODEL WITH BANKING DATASET

## 1. Tổng quan

Đồ án thực hiện **Instruction Fine-tuning** mô hình `Qwen3-4B-Base` để phân loại 77 loại ý định của khách hàng trong lĩnh vực ngân hàng. Thay vì huấn luyện lại toàn bộ mô hình, em sử dụng **LoRA** — chỉ thêm một lượng nhỏ tham số có thể học — giúp giảm đáng kể bộ nhớ GPU và thời gian huấn luyện. Thư viện **Unsloth** tối ưu thêm tốc độ forward/backward pass lên đến 2x so với cài đặt gốc.

| Thành phần        | Chi tiết                        |
| ------------------- | -------------------------------- |
| Base Model          | `unsloth/Qwen3-4B-Base`        |
| Phương pháp      | LoRA (PEFT) + Instruction Tuning |
| Thư viện training | Unsloth + TRL (SFTTrainer)       |
| Tập dữ liệu      | Banking77 (77 classes)           |
| Môi trường       | Google Colab                   |

## 2. Cấu trúc thư mục

```
banking-intent-unsloth/
│
├── configs/
│   ├── train.yaml          # Cấu hình hyperparameter huấn luyện
│   └── inference.yaml      # Cấu hình đường dẫn model cho inference
│
├── scripts/
│   ├── preprocess_data.py  # Tiền xử lý và chia tách dữ liệu
│   ├── train.py            # Script huấn luyện chính
│   ├── inference.py        # Class IntentClassification cho inference
│   └── evalute.py          # Đánh giá base model và fine-tuned model
│
├── sample_data/
│   ├── train.csv           # Tập huấn luyện
│   ├── test.csv            # Tập kiểm tra
│   └── labels.csv          # Danh sách nhãn (77 intents)
│
├── train.sh                # Shell script chạy pipeline huấn luyện
├── inference.sh            # Shell script chạy inference
├── requirements.txt        # Danh sách thư viện cần cài đặt
└── README.md

```


## 3. Tập dữ liệu

**Banking77** là benchmark phân loại ý định phổ biến trong lĩnh vực ngân hàng:

| Thông số        | Giá trị                                    |
| ----------------- | -------------------------------------------- |
| Số lớp          | 77 intents                                   |
| Ngôn ngữ        | Tiếng Anh                                   |
| Tập huấn luyện | **10.003 mẫu** (toàn bộ tập train) |
| Tập kiểm tra    | **3.080 mẫu** (toàn bộ tập test)   |
| Định dạng      | CSV (`text`, `label`, `label_text`)    |

**Ví dụ dữ liệu:**

| text                                   | label | label_text            |
| -------------------------------------- | ----- | --------------------- |
| "I'm not sure why my card didn't work" | 25    | declined_card_payment |
| "limits on top ups"                    | 60    | top_up_limits         |
| "Why did my top-up not work?"          | 59    | top_up_failed         |

**Prompt Template:**

```
Below is an inquiry from a bank customer. Classify the intent of this message.

### Instruction:
Classify the following message into one of the banking categories.

### Input:
{câu hỏi của khách hàng}

### Response:
{nhãn intent}
```

## 4. Các Hyperparameter

Tất cả hyperparameter được quản lý tập trung tại file [`configs/train.yaml`](configs/train.yaml).

### 4.1 Cấu hình Mô hình

| Tham số           | Giá trị                 | Mô tả                                         |
| ------------------ | ------------------------- | ----------------------------------------------- |
| `model.name`     | `unsloth/Qwen3-4B-Base` | Tên model gốc tải từ HuggingFace/Unsloth    |
| `max_seq_length` | `256`                   | Độ dài tối đa của chuỗi token đầu vào |
| `num_labels`     | `77`                    | Số lớp intent của tập Banking77             |
| `load_in_4bit`   | `True`                  | Lượng tử hoá 4-bit để tiết kiệm VRAM    |

### 4.2 Cấu hình LoRA (PEFT)

| Tham số                  | Giá trị                                                         | Mô tả                                                           |
| ------------------------ | --------------------------------------------------------------- | --------------------------------------------------------------- |
| `lora.r`                 | `16`                                                            | Rank của ma trận LoRA — cân bằng giữa khả năng học và tham số   |
| `lora.alpha`             | `32`                                                            | Hệ số scaling LoRA                                              |
| `lora.dropout`           | `0.05`                                                          | Dropout áp dụng lên LoRA layers để tránh overfitting            |
| `target_modules`         | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Các module được gắn LoRA adapter                                |
| `bias`                   | `none`                                                          | Không huấn luyện bias terms                                     |
| `gradient_checkpointing` | `unsloth`                                                       | Tối ưu bộ nhớ với unsloth gradient checkpointing                |

### 4.3 Cấu hình Huấn luyện

| Tham số                  | Giá trị      | Mô tả                                                                        |
| -------------------------- | -------------- | ------------------------------------------------------------------------------ |
| `batch_size`             | `16`         | Số mẫu xử lý mỗi bước trên mỗi GPU                                    |
| `learning_rate`          | `2e-4`       | Tốc độ học — phù hợp với LoRA fine-tuning                              |
| `epochs`                 | `3`          | Số vòng lặp qua toàn bộ tập huấn luyện                                 |
| `optimizer`              | `adamw_8bit` | AdamW lượng tử hoá 8-bit — tiết kiệm VRAM, hiệu năng tương đương |
| `weight_decay`           | `0.01`       | L2 regularization, giúp tránh overfitting                                    |
| `warmup_steps`           | `10`         | Số bước learning rate tăng dần từ 0 lên `learning_rate`               |
| `lr_scheduler_type`      | `cosine`     | Cosine annealing — giảm learning rate theo hình cosine                      |
| `seed`                   | `3407`       | Random seed để tái hiện kết quả                                          |
| `eval_strategy`          | `epoch`      | Đánh giá trên tập test sau mỗi epoch                                     |
| `save_strategy`          | `epoch`      | Lưu checkpoint sau mỗi epoch                                                 |
| `load_best_model_at_end` | `True`       | Tự động load checkpoint tốt nhất sau khi training kết thúc              |
| `fp16` / `bf16`          | Tự động      | Mixed precision — fp16 hoặc bf16 tùy phần cứng GPU                        |

### 4.4 Cấu hình Inference

| Tham số            | Giá trị                           | Mô tả                                       |
| ------------------- | ----------------------------------- | --------------------------------------------- |
| `checkpoint_path` | `checkpoints/qwen_banking_intent` | Đường dẫn đến model đã fine-tune      |
| `max_seq_length`  | `256`                             | Khớp với max_seq_length lúc training |

## 5.Triển khai trên Google Colab

### Bước 1 — Clone repository và di chuyển vào thư mục làm việc

```python
!git clone https://github.com/ZelusIsMe-05/banking-intent-unsloth.git
%cd banking-intent-unsloth
```

### Bước 2 — Kết nối Google Drive (để lưu/lấy checkpoint)

```python
from google.colab import drive
import shutil
import os

print("Mounting Google Drive...")
drive.mount('/content/drive')
```

**Lưu ý:** Một cửa sổ xác thực sẽ hiện ra — làm theo hướng dẫn để cho phép Colab truy cập Google Drive của bạn.

### Bước 3 — Cài đặt các thư viện cần thiết

```python
!pip install -r requirements.txt
```

### Bước 4 — Huấn luyện mô hình

```python
!bash train.sh
```

Pipeline này sẽ thực hiện tuần tự:

1. **Tiền xử lý dữ liệu** (`scripts/preprocess_data.py`) — chuẩn bị file CSV train/test.
2. **Fine-tuning mô hình** (`scripts/train.py`) — áp dụng LoRA và huấn luyện với các hyperparameter trong `configs/train.yaml`.

Checkpoint được lưu vào `checkpoints/qwen_banking_intent/` sau mỗi epoch.

### Bước 5 — Sao chép checkpoint lên Google Drive

Chạy cell này ngay sau khi training hoàn tất để tránh mất dữ liệu khi Colab timeout:

```python
source_dir = "/content/banking-intent-unsloth/checkpoints"
dest_dir = "/content/drive/MyDrive/banking-intent-unsloth-checkpoints"

print(f"Copying model checkpoints to: {dest_dir}")
try:
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
    print("Successfully saved checkpoints to Google Drive!")
except Exception as e:
    print(f"An error occurred: {e}")
```

Sau bước này, toàn bộ checkpoint được lưu an toàn tại `MyDrive/banking-intent-unsloth-checkpoints/`.

### Bước 6 — Khôi phục checkpoint từ Google Drive (khi mở lại Colab)

Nếu session Colab bị ngắt và cần file **checkpoints** đã lưu ở trên, thì cần lấy trên drive đã lưu như sau:

```python
import shutil

drive_source_dir = "/content/drive/MyDrive/banking-intent-unsloth-checkpoints"
colab_dest_dir = "/content/banking-intent-unsloth/checkpoints"

print(f"Loading model checkpoints from Drive to: {colab_dest_dir}")
try:
    shutil.copytree(drive_source_dir, colab_dest_dir, dirs_exist_ok=True)
    print("✅ Successfully loaded checkpoints from Google Drive back to Colab!")
except Exception as e:
    print(f"❌ An error occurred: {e}")
```

### Bước 7 — Chạy Inference (Demo thử nghiệm)

```python
!bash inference.sh
```

Lúc này, chương trình cho phép nhập câu hỏi bắt kỳ về chủ đề ngân hàng và nhận kết quả phân loại intent ngay lập tức.

**Ví dụ:**

```
Enter a message: I want to block my credit card
Predicted Intent: card_not_working

Enter a message: 'Card swallowed by ATM'
Predicted Intent: card_swallowed
```

### Bước 8 — Đánh giá mô hình

```python
!python scripts/evalute.py
```

Script này sẽ:

1. Đánh giá **Fine-tuned Model** (sau khi áp dụng LoRA) trên tập test.
2. Lưu kết quả vào `checkpoints/eval_results.json`.

**Kết quả đầu ra mẫu:**

```json
{
    "metrics": {
        "total_samples": 3080,
        "fine_tuned_correct": 2841,
        "fine_tuned_accuracy": 0.9224025974025974
    }
}
```

## 6. Các liên kết liên quan

Thầy có thể truy cập link chứa video demo được em gắn ở đây: [VIdeo demo tại đây](https://drive.google.com/file/d/13u1VI15q0UhkXsafeG5Z8ccin0WMILVz/view?usp=sharing).

Thư mục chứa checkpoints của em sau khi được huấn luyện: [Checkpoints tại đây](https://drive.google.com/drive/folders/1-j5gcJaPAiBqyduRS87lcW0UwJfL02k2?usp=sharing).
