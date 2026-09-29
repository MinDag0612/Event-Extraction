# Best checkpoints OneIE đã lưu

Đã tải đủ bốn checkpoint được chọn theo dev Argument-C. Đã mở trên CPU, kiểm tra trọng số hữu hạn và đối chiếu SHA256 với manifest tải về. Chưa chạy lại inference từ checkpoint.

| Dataset | Epoch chọn theo log | Dung lượng (MiB) | File |
|---|---:|---:|---|
| BKEE | 30 | 693.43 | [best.role.mdl](../.kaggle-deploy/archive/BKEE/oneie-bkee/runs/20260920_141457/best.role.mdl) |
| VHE | 27 | 693.23 | [best.role.mdl](../.kaggle-deploy/archive/VHE/oneie-vhe/runs/20260923_172443/best.role.mdl) |
| PHEE | 13 | 692.74 | [best.role.mdl](../.kaggle-deploy/archive/PHEE/oneie-phee/runs/20260923_172040/best.role.mdl) |
| GENEVA | 29 | 695.59 | [best.role.mdl](../.kaggle-deploy/archive/GENEVA/oneie-geneva/runs/20260924_031951/best.role.mdl) |

Thư mục gốc: `.kaggle-deploy/archive/`. Config ở `DATASET/oneie-dataset/config.json`; log và prediction dev/test cùng thư mục với `best.role.mdl`. Mỗi checkpoint cũng chứa config, vocabulary và valid patterns.

[SHA256 và kiểm tra checkpoint](best_checkpoints.json) · [Kiểm chứng điểm số và checkpoint](verification.json).

Đây là bốn best checkpoint, không phải checkpoint của từng epoch. Không xác nhận lưu đủ `last.mdl` để resume optimizer/scheduler. Các phần tải dở `.part`/`.chunks`, ZIP checkpoint cũ và bản output trùng đã được dọn. Log 11 epoch đầu BKEE giữ tại `.kaggle-deploy/archive/BKEE/history/` để truy nguyên lượt resume. Thư mục `.kaggle-deploy` bị gitignore; commit báo cáo không sao lưu các model này sang máy khác.
