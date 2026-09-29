# Kết quả baseline OneIE + mBERT

Đây là nơi lưu số liệu chính của baseline OneIE. Điểm dev/test tại checkpoint được chọn đã được chấm lại; bốn best checkpoint đã tải đủ và mở trên CPU. Số liệu không thay đổi sau kiểm chứng.

## Mở file nào?

- Đọc/nộp báo cáo: [REPORT.md](../reports/oneie_four_datasets/REPORT.md).
- Lấy bảng so với SLM: [final_results.csv](final_results.csv).
- Phân tích 30 epoch: [epoch_metrics.csv](epoch_metrics.csv).
- Kiểm tra dữ liệu chuyển đổi: [data_statistics.csv](data_statistics.csv).
- Tìm model/config/prediction: [CHECKPOINTS.md](CHECKPOINTS.md).
- Xem bằng chứng kiểm chứng: [verification.json](verification.json).

Không còn bản CSV epoch trùng trong `reports/`; dùng `results/epoch_metrics.csv` làm nguồn bảng điểm chính.

| File | Nội dung |
|---|---|
| `epoch_metrics.csv` | 120 dòng: 4 dataset × 30 epoch, P/R/F1 dev và test |
| `final_results.csv` | 7 dòng: BKEE theo upstream; mỗi bộ VHE/PHEE/GENEVA có 2 phạm vi gold |
| `data_statistics.csv` | 12 dòng: số câu và annotation của train/dev/test trên 4 bộ |
| `source_manifest.json` | Đường dẫn và SHA256 của các file thực tế dùng để xuất CSV |

## Quy ước đọc số liệu

- Epoch đánh số **1–30**; `epoch_index` giữ index 0–29 của log.
- P/R/F1 và `retention` đều dùng thang **0–100**, không phải 0–1. Không làm tròn khi xuất CSV.
- `trig_i`, `trig_c`, `arg_i`, `arg_c`: trigger identification/classification và argument identification/classification. Mỗi nhóm có cột riêng `_p`, `_r`, `_f1`.
- `train_loss` để trống vì log epoch đã lưu không có trường này. Trống là **thiếu số liệu**, không phải loss bằng 0. `train_loss_status` ghi lý do; muốn bổ sung phải trích được loss thật có định nghĩa tổng hợp rõ ràng.
- Scorer argument OneIE dùng span argument + event type (+ role cho classification), không bắt buộc khớp vị trí trigger. Cần dùng cùng scorer khi so với luồng chính.

## Chọn kết quả cuối

Checkpoint được chọn bằng dev Argument-C F1 cao nhất; hòa thì lấy epoch sớm hơn. Test chỉ phục vụ báo cáo, không chọn checkpoint.

`gold_protocol` phân biệt:

- `upstream_bkee`: giao thức gold/scorer của lượt BKEE cũ.
- `projected`: annotation đã lọc overlap/multirole để phù hợp OneIE.
- `unpruned_token_aligned`: annotation trước lọc, nhưng đã căn chỉnh token; PHEE đã tách fragment. Không gọi đây là raw gold nguyên trạng hay scorer PHEE chính thức.

Hai dòng của mỗi bộ mới dùng **cùng checkpoint**, không phải hai lượt train. `selection_dev_arg_c_f1` là điểm thực tế dùng chọn checkpoint; `dev_arg_c_f1` là điểm dev theo gold của dòng đó. Chọn dòng theo `gold_protocol` khi lập bảng so sánh, không gộp trung bình 7 dòng.

Điểm trước lọc chỉ có tại checkpoint đã chọn, không có đủ 30 epoch. BKEE có 11 epoch đầu và 19 epoch weights-only resume, optimizer/scheduler khởi tạo lại. Các cột `training_phase` và `resume_mode` giữ thông tin này.

## Thống kê dữ liệu

`retention = argument_after / argument_before × 100`. Dùng `dropped_arguments` và `dropped_events` riêng thay cho `dropped_annotations` chung, tránh cộng lẫn các loại annotation.

Với ba bộ mới, before/after là trước/sau projection trên token-aligned gold; PHEE đã tách fragment trước khi đếm. Đây là số liên kết annotation, không phải mẫu số sau khử trùng lặp của scorer.

Với BKEE, thống kê chỉ phản ánh **bước preparation** giữ annotation từ raw, nên retention là 100%. Không được suy ra xử lý graph/overlap upstream cũng giữ 100%. Cột `processing_stage` đánh dấu khác biệt này.

## Nguồn và tái lập

```bash
python scripts/export_oneie_results.py
```

Cần có `reports/oneie_four_datasets/evidence/` và raw BKEE tại máy để tạo lại thống kê. Script không đọc token hay gọi Kaggle. CSV không thay thế việc sao lưu checkpoint và prediction; việc xuất CSV không xác nhận các artifact đó đã tải đủ.

Xem [báo cáo phân tích](../reports/oneie_four_datasets/REPORT.md) để đọc mapping, split và giới hạn thực nghiệm. VHE dùng split tự tạo seed 42; các phương pháp so sánh cần dùng cùng manifest.

## Kết quả kiểm chứng prediction

[verification.json](verification.json) ghi nhận `SCORES_AND_BEST_CHECKPOINT_PASS` cho cả bốn bộ. Đã chấm lại prediction dev/test tại checkpoint được chọn: toàn bộ P/R/F1 khớp log với sai lệch lớn nhất 0.0. Điểm trước lọc của ba bộ mới cũng khớp 0.0; ID/token/span hợp lệ và không thiếu câu. Vì vậy **không thay đổi số liệu CSV** sau kiểm chứng.

Các epoch khác vẫn là số liệu log, chưa được chấm lại từng epoch. Đã tải và mở thành công bốn best checkpoint trên CPU, trọng số hữu hạn; xem [vị trí và bằng chứng](CHECKPOINTS.md). Chưa chạy inference lại từ trọng số. Chấm lại bằng cùng scorer kiểm tra tính nhất quán kết quả, không chứng minh scorer không có giới hạn; khi so SLM phải giữ cùng split, gold và quy tắc chấm.

```bash
python scripts/verify_oneie_archive.py --scores-only
```
