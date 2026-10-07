# OneIE baseline trên curated-v2

## Trạng thái

Đã chuẩn bị dữ liệu, runner và bốn notebook Kaggle. Notebook trong repository mặc định chỉ kiểm tra.
Ngày 2026-10-07, người dùng đã cho phép triển khai cả bốn dataset. Đã upload dataset private
`vietquan299/oneie-curated-v2` và gửi job BKEE/GENEVA (version 2, sửa nhận ZIP được Kaggle giải nén).
Mỗi job chạy smoke trước, chỉ thành công mới tự chạy full 30 epochs. Chưa có điểm baseline mới.
PHEE/VHE nằm trong hàng đợi vì tài khoản giới hạn 2 batch GPU sessions.

Theo dõi trạng thái thực tế ở `.kaggle-deploy/curated-v2-runs/queue-state.json` và `queue.log`.
`scripts/queue_oneie_kaggle.py` tự gửi job chờ khi có slot, tải báo cáo khi hoàn tất và dừng
hàng đợi khi job lỗi. Tiến trình chạy trên máy local; cần giữ máy bật và có mạng.
Job đã gửi lên Kaggle tiếp tục chạy độc lập với máy local. Checkpoint giữ trên Kaggle;
hàng đợi chỉ tải JSON/JSONL/log. Không tự retry job lỗi hoặc ghi đè kết quả cũ.

| Dataset | TRAIN | DEV | TEST |
| --- | ---: | ---: | ---: |
| BKEE | 9341 | 4039 | 3728 |
| GENEVA | 1874 | 766 | 930 |
| PHEE | 2853 | 949 | 961 |
| VHE | 3267 | 406 | 397 |

29.511 mẫu thuộc manifest curated chung. Dữ liệu nguồn có 65 mẫu bị quarantine
vì lệch span và 2.042 mẫu bị loại theo chính sách trùng/xung đột nhãn.
Các kiểm tra kỹ thuật không đảm bảo mọi nhãn ngữ nghĩa đều đúng.

## Có dùng lại kết quả cũ không?

Giữ nguyên kết quả và checkpoint cũ để lưu lịch sử, ghi rõ bộ chia, cách chuẩn bị và
scorer cũ. Không đưa các điểm đó vào cùng bảng so sánh chính với SLM trên curated-v2.
Bản cũ có câu trùng TRAIN–TEST (BKEE: 807, GENEVA: 20), VHE dùng bộ chia khác,
và xử lý span/gold khác. Chấm lại checkpoint cũ không loại được ảnh hưởng dữ liệu
mà mô hình đã học. Baseline mới cần khởi tạo từ encoder pretrained và train lại.
Không dùng notebook `OneIE_BKEE_Resume_Kaggle.ipynb` cho thí nghiệm mới; đó là workflow cũ.

## Giao thức so sánh

- Cùng manifest TRAIN/DEV/TEST, ID, text, tokens và gold đầy đủ cho OneIE và SLM.
- OneIE cần projection khi train: không biểu diễn được mọi span chồng lấn hoặc
  nhiều role trên cùng cạnh. Gold đầy đủ vẫn được giữ riêng để đánh giá. Vì vậy
  TRAIN F1 gần 1 không phải điều kiện khả thi cho mọi dataset/model.
- TI: trigger span; TC: trigger span + event type.
- AI: event type + argument span; AC: AI + role. Micro F1 theo tập nhãn từng mẫu.
  Argument không yêu cầu khớp vị trí trigger. Các đoạn rời được chấm riêng;
  chưa chấm khả năng nối đoạn thành cùng mention. Không gọi điểm này là đánh giá
  đầy đủ grouping của PHEE. Nếu cần, bổ sung metric chung trước khi chạy cả hai mô hình.
- Chọn checkpoint bằng DEV AC trên gold đầy đủ, giữ epoch sớm nhất khi bằng điểm.
  Chỉ suy luận TEST sau khi chọn checkpoint. Lưu điểm TRAIN/DEV/TEST của checkpoint đó.
- mBERT pinned revision; seed 42; 30 epochs; batch hiệu dụng 8; beam 5.
  Tắt global features; cho phép mọi cặp event-role trong vocabulary.
  Đây là cấu hình OneIE của thí nghiệm này, không phải tái lập nguyên cấu hình cũ.
- Vocabulary lấy từ TRAIN; GENEVA bổ sung catalog ontology chính thức đã khóa revision,
  không thu thập label từ gold DEV/TEST để tạo vocabulary.
- Một seed đủ tạo một kết quả baseline; muốn kết luận ổn định nên chạy thêm các seed
  đã chọn trước cho cả hai mô hình và báo cáo mean/std.

## Quy trình chạy sau khi kiểm tra

1. ZIP đã chuẩn bị tại `.kaggle-deploy/curated-v2-upload/oneie-curated-v2.zip`.
   File `.sha256.json` cạnh đó ghi checksum. ZIP chứa dữ liệu và mã cần thiết,
   không chứa credentials, checkpoint cũ hoặc pretrained weights.
2. Sau khi duyệt, tự attach ZIP lên Kaggle; mở một trong bốn notebook
   `notebooks/baselines/OneIE_{BKEE,GENEVA,PHEE,VHE}_Kaggle.ipynb`.
   Sửa `BUNDLE` theo đường dẫn Kaggle thực tế. `Qwen_model.ipynb` không phải entrypoint OneIE.
3. Giữ `RUN_MODE='check'`, `CONFIRM_RUN=False` để kiểm tra checksum và dữ liệu.
   Nếu thiếu dependencies, bật `INSTALL_DEPENDENCIES`. PyTorch/CUDA do môi trường Kaggle cung cấp.
   Các phiên bản trong requirements đã dùng cho kiểm tra dữ liệu local; chưa kiểm chứng training GPU.
4. Sau khi duyệt, bật GPU/Internet để tải encoder; đặt `RUN_MODE='smoke'`,
   `CONFIRM_RUN=True`. Smoke chạy 1 epoch trên tập nhỏ, kiểm tra train, decode,
   lưu/nạp checkpoint và scorer. Điểm smoke không dùng làm baseline.
5. Smoke thành công mới đặt `RUN_MODE='full'`, trỏ `SMOKE_RUN` tới thư mục smoke còn tồn tại.
   Nếu đổi phiên Kaggle, lưu/attach lại thư mục smoke. Runner kiểm tra checksum tương thích.
   Chọn thư mục output mới; không ghi đè run cũ. Full không tự chạy ngay sau smoke.
6. Lưu toàn bộ thư mục run: `run.json`, `config.json`, `epochs.jsonl`, `best.pt`,
   `predictions.{train,dev,test}.jsonl`, `scores.json`, `completed.json`.
   Chỉ dùng điểm full khi có completed.json, đủ ID và không có lỗi/nonfinite loss.

SLM phải xuất Unified predictions đủ mọi ID, kể cả mẫu dự đoán không có event,
và dùng cùng scorer:

```sh
python -m scripts.score_oneie_predictions --predictions predictions.test.jsonl \
  --gold data/processed/oneie-ready-v2/PHEE/gold/test.jsonl --output slm-test-scores.json
```

## Kiểm tra local không train

Dùng Python có PyTorch và transformers (môi trường `.venv` hiện tại thiếu PyTorch):

```sh
python3 -m unittest discover -s tests -p 'test_oneie_curated.py' -v
python3 -m scripts.check_oneie_ready \
  --oneie .kaggle-deploy/curated-oneie-source-v2 --local-files-only
python3 -m scripts.bundle_oneie --output /tmp/oneie-curated-v2-new.zip
```

`check_oneie_ready` numberize toàn bộ 29.511 mẫu và collate một batch mỗi split,
không tạo mô hình. ZIP cũng được kiểm tra trong thư mục tạm độc lập để phát hiện
thiếu module. Các bước này chưa thay thế smoke test GPU.


## Điều chỉnh sau triển khai (2026-10-07)

GENEVA version 2 dừng bằng SIGKILL trong smoke. Kiểm tra mã tìm thấy OneIE vẫn tạo
bảng global feature dù `use_global_features=False`; vocabulary GENEVA làm bảng
kết hợp event-role phát triển rất lớn. Source patch 2 bỏ cấp phát bảng khi đã tắt.
Đã kiểm tra khởi tạo bằng vocabulary GENEVA thực với encoder nhỏ ngẫu nhiên;
không gọi hàm tạo global maps. Dataset Kaggle version 2 chứa patch này,
giữ nguyên dữ liệu và scorer. GENEVA gửi lại với patch 2; PHEE/VHE chờ dùng patch 2.
BKEE đã gửi trước vẫn dùng patch 1; sửa đổi chỉ loại bộ nhớ không sử dụng khi global
features tắt, không đổi thuật toán forward, gold hay cách chấm.

### So sánh số epoch với SLM

OneIE tối đa 30 epochs; SLM dự kiến các mốc 1/3/5/7/10. Mỗi bên chọn bằng DEV,
sau đó dùng TEST cho checkpoint đã chọn. Báo cáo rõ phạm vi epoch, epoch được chọn,
seed và cấu hình. Đây là so sánh chất lượng với ngân sách huấn luyện khác nhau;
không diễn giải là so sánh cùng lượng tính toán. So sánh chi phí cần thêm thời gian,
GPU và số tham số trainable. Không chọn epoch hoặc điều chỉnh cấu hình theo TEST.

## Đối chiếu nhánh metrics/all_score

Đã đọc `origin/metrics/all_score` tại commit `4ecefe9`:

| Metric | Nhánh metrics/all_score | Runner OneIE hiện tại |
| --- | --- | --- |
| TI | trigger span | trigger span |
| TC | trigger span + event type | trigger span + event type |
| AI | argument span | argument span + event type |
| AC | argument span + role | argument span + role + event type |

Không trộn điểm AI/AC của hai định nghĩa trong cùng bảng so sánh. Nhánh metrics
có `evaluate_on_sample` tích lũy counts, nhưng `evaluate_on_file` và
`evaluate_on_all_sample` còn `pass`; notebook `test_metrics.ipynb` minh họa mẫu nhỏ.
Để chấm toàn bộ TRAIN/DEV/TEST cần ghép đủ prediction–gold theo ID, kiểm tra coverage,
khởi tạo evaluator mới cho mỗi split và lấy micro score sau khi cập nhật hết mẫu.
Chưa đổi scorer hoặc checkpoint selection của job Kaggle đang chạy.
Nếu đổi scorer chính, cần thống nhất lại cả tiêu chí chọn checkpoint trên DEV;
chấm lại TEST đơn thuần không tương đương với chọn lại checkpoint bằng scorer mới.
