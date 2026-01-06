# Rice Leaf Disease Detection - Backend

Hệ thống phát hiện các triệu chứng thiếu dinh dưỡng trên lá lúa sử dụng Deep Learning.

## 📋 Mô tả

Backend API phục vụ việc phát hiện các triệu chứng thiếu chất dinh dưỡng N (Nitrogen), P (Phosphorus), K (Potassium) trên lá lúa thông qua hình ảnh. Hệ thống hỗ trợ nhiều mô hình Deep Learning khác nhau để đảm bảo độ chính xác cao.

### Các mô hình được hỗ trợ:

- **Xception GOC** - Xception architecture từ thư viện timm
- **MiniXception** - Lightweight custom Xception architecture
- **Xception ECA** - Xception với Efficient Channel Attention mechanism
- **EfficientNetB0** - EfficientNet-B0 architecture
- **MobileNetV3** - MobileNet-V3-Large architecture

## 📊 Dataset

Dataset: [Nutrient Deficiency Symptoms in Rice](https://www.kaggle.com/datasets/guy007/nutrientdeficiencysymptomsinrice)

Dataset bao gồm các hình ảnh lá lúa với 3 loại thiếu chất:

- Nitrogen (N) deficiency
- Phosphorus (P) deficiency
- Potassium (K) deficiency

## 🛠️ Yêu cầu hệ thống

- Python 3.8+
- uv (khuyến nghị) hoặc pip
- CUDA (optional, cho GPU acceleration)

## 📦 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd demo/be
```

### 2. Cài đặt uv (nếu chưa có)

**Windows:**

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/Mac:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Hoặc dùng pip:

```bash
pip install uv
```

### 3. Tạo môi trường ảo và cài dependencies

**Sử dụng uv (khuyến nghị):**

```bash
# uv tự động tạo virtual environment và cài dependencies
uv sync
```

**Hoặc sử dụng pip truyền thống:**

```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt environment
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

### 4. Cấu hình environment variables

Copy file `.env.example` thành `.env.local`:

```bash
# Windows
copy .env.example .env.local

# Linux/Mac
cp .env.example .env.local
```

Sau đó chỉnh sửa file `.env.local`

**Cấu hình:**

1. **DATABASE_URL**: Đường dẫn đến SQLite database

   - Mặc định: `sqlite:///./history.db` (tạo file history.db trong thư mục be/)
   - File sẽ được tự động tạo khi chạy ứng dụng lần đầu

2. **GOOGLE_API_KEY**: API key cho Gemini AI (cung cấp lời khuyên)
   - Truy cập [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Tạo API key mới
   - Copy và paste vào file `.env.local`

### 5. Download model weights

Các file model weights cần được đặt trong thư mục `models/`:

- `best_xception_goc_overall.pth`
- `best_minixception_overall.pth`
- `best_xception_eca.pth`
- `best_efficientnetb0_overall.pth`
- `best_mobilenetv3_overall.pth`

## 🚀 Chạy ứng dụng

### Chế độ Development

**Sử dụng uv:**

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Hoặc trực tiếp với Python:**

```bash
# Kích hoạt virtual environment trước (nếu dùng pip)
python main.py
```

### Chế độ Production

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server sẽ chạy tại: `http://localhost:8000`

## 📚 API Documentation

Sau khi chạy server, truy cập:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🗄️ Database

Hệ thống sử dụng SQLite để lưu trữ lịch sử dự đoán.

### Migration với Alembic

**Tạo migration mới:**

```bash
alembic revision --autogenerate -m "description"
```

**Chạy migration:**

```bash
alembic upgrade head
```

**Rollback migration:**

```bash
alembic downgrade -1
```

## 🧪 Training Models

Các notebook training được lưu trong thư mục `notebooks/`:

- `final_x-cus.ipynb` - Training MiniXception
- `final_xception-goc.ipynb` - Training Xception GOC
- `minixception-attention-eca.ipynb` - Training Xception ECA
- `efficientnet-b0.ipynb` - Training EfficientNetB0
- `mobilenetv3.ipynb` - Training MobileNetV3

## 📝 License

MIT License
