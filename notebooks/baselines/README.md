# Chạy OneIE baseline trên Kaggle

Bốn notebook hiện tại:

- `OneIE_BKEE_Kaggle.ipynb`
- `OneIE_GENEVA_Kaggle.ipynb`
- `OneIE_PHEE_Kaggle.ipynb`
- `OneIE_VHE_Kaggle.ipynb`

## Không cần gửi ZIP thủ công

1. Import notebook tương ứng từ GitHub vào Kaggle, hoặc copy notebook Kaggle được chia sẻ.
2. Trong **Add Input**, gắn dataset `vietquan299/oneie-curated-v2`, version 2
   (chứa OneIE source patch 2). Đây là dataset private: tài khoản chạy phải có quyền truy cập.
   Copy notebook không tự cấp quyền cho dataset private.
3. Bật Internet và chọn GPU. Đặt `INSTALL_DEPENDENCIES=True` nếu cần cài thư viện.
4. Mặc định `RUN_MODE='check'`, `CONFIRM_RUN=False`: chạy các kiểm tra checksum/đầu vào.
5. Đặt `RUN_MODE='smoke'`, `CONFIRM_RUN=True`: kiểm tra một lượt train nhỏ và lưu/nạp checkpoint.
6. Khi smoke hoàn tất, đặt `RUN_MODE='full'`, giữ thư mục `SMOKE_RUN` và chạy cell thực thi.
   Full dùng tối đa 30 epochs và chọn checkpoint bằng DEV AC F1.
   Nếu đổi phiên Kaggle, cần lưu/attach lại thư mục smoke và chỉnh `SMOKE_RUN`.

Notebook tự nhận dữ liệu Kaggle đã giải nén hoặc dạng ZIP. Các module runner, scorer,
OneIE và đầu vào đã chuẩn bị nằm trong input dataset; không cần clone repository để chạy.

## Giới hạn hiện tại

Notebook vẫn phụ thuộc input dataset; chưa tự tải dữ liệu gốc và tái tạo curated-v2.
Để bỏ phụ thuộc này cần bổ sung pipeline dựng lại dữ liệu và xác minh checksum khớp
manifest đã khóa. Không thay bằng dữ liệu gốc chưa lọc hoặc tự chia lại split nếu muốn
so sánh trực tiếp với baseline hiện tại.

`OneIE_BKEE_Resume_Kaggle.ipynb` là workflow cũ, không dùng cho baseline mới.
Chi tiết protocol: [ONEIE_CURATED.md](../../docs/ONEIE_CURATED.md).
