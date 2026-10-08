# Curated-v2: gold chung cho OneIE và SLM


| Dataset | Train | Dev | Test |
|---|---:|---:|---:|
| BKEE | 9341 | 4039 | 3728 |
| GENEVA | 1874 | 766 | 930 |
| PHEE | 2853 | 949 | 961 |
| VHE | 3267 | 406 | 397 |

Mỗi dòng chứa `id`, `text`, `tokens`, `events`. Dùng `text` làm input và chuyển
`events` thành target của SLM. Span là chỉ số token trong `tokens`, bắt đầu từ 0,
đầu đóng/cuối mở; không phải vị trí subword của tokenizer SLM. Giữ metadata nhóm
mention rời của PHEE khi tạo target nếu cần giữ đầy đủ cấu trúc.

Không chia lại split hoặc tự bỏ mẫu. Chọn checkpoint bằng DEV AC; chỉ dùng TEST
đánh giá sau khi chọn. Scorer chung: TI=trigger span; TC=trigger span+event type;
AI=argument span+event type; AC=argument span+event type+role, micro set matching
trên từng mẫu, chấm từng đoạn mention; chưa yêu cầu khớp vị trí trigger cho AI/AC.

`manifest.json` lưu SHA-256, số mẫu và kích thước của từng file. Việc tải lại dữ
liệu gốc và chạy adapters chưa đảm bảo tái tạo đúng bộ lọc/split này; dùng các file
đã khóa ở đây để so sánh cùng baseline. Quyền sử dụng của các dataset gốc vẫn áp dụng.


