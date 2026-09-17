# Hướng dẫn sử dụng tools

Thư mục này chứa các script hỗ trợ cho project. Mỗi tool nên có một phần hướng dẫn ngắn trong file này để sau này dễ bổ sung thêm.

## 1. Chuẩn bị Kaggle token

Tool `upload_to_kaggle.py` cần quyền truy cập Kaggle để tạo dataset.

### Cách khuyến nghị: đăng nhập bằng Kaggle CLI

Mở PowerShell tại thư mục project và chạy:

```powershell
uv add kaggle
uv run kaggle auth login
```

Kaggle sẽ yêu cầu nhập thông tin đăng nhập hoặc API token. Làm theo hướng dẫn hiển thị trên màn hình.

### Cách dùng file `kaggle.json`

1. Vào Kaggle: https://www.kaggle.com/settings/account
2. Tìm mục **API** và chọn **Create New Token**.
3. File `kaggle.json` sẽ được tải xuống.
4. Đặt file vào thư mục sau trên Windows:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

Ví dụ:

```text
C:\Users\<ten-user>\.kaggle\kaggle.json
```

Không đưa file token lên Git hoặc chia sẻ file này với người khác. Nếu token bị lộ, hãy xóa token cũ trên Kaggle và tạo token mới.

## 2. Kiểm tra đăng nhập

Chạy:

```powershell
uv run kaggle auth status
```

Nếu lệnh hiển thị username hoặc trạng thái đăng nhập thành công, có thể dùng tool.

## 3. Upload dataset hoặc model lên Kaggle

Đứng ở thư mục gốc của project rồi chạy:

```powershell
uv run python tools/upload_to_kaggle.py
```

Tool sẽ hỏi lần lượt:

1. **Đường dẫn thư mục chứa dataset/model**: đường dẫn tới thư mục cần upload.
2. **Tên khi lên Kaggle**: tên hiển thị của dataset; mặc định là tên thư mục.
3. **Kaggle username**: tool tự đọc từ trạng thái đăng nhập, nếu không đọc được thì nhập thủ công.
4. **License**: có thể để trống nếu không muốn khai báo.
5. **Xác nhận upload**: nhập `y` hoặc `yes` để bắt đầu upload.

Dataset được tạo với dạng private và có ID theo mẫu:

```text
<kaggle-username>/<slug>
```

Ví dụ, nếu tên là `My Event Dataset`, ID có thể là:

```text
my-kaggle-name/my-event-dataset
```

Tool tạo một bản sao tạm thời và thêm `dataset-metadata.json` vào bản sao đó. File gốc của dataset/model không bị thêm hoặc chỉnh sửa metadata.

## 4. Ví dụ chạy

```powershell
uv run python tools/upload_to_kaggle.py
```

Khi được hỏi đường dẫn, có thể nhập:

```text
D:\EVENT EXTRACTION\sourse\data\processed\GENEVA
```

Nếu đường dẫn có dấu cách, có thể nhập cả đường dẫn trong dấu ngoặc kép.
