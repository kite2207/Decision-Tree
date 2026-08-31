# Thí nghiệm cải thiện cây quyết định

Đã hoàn thành ba thí nghiệm trong phần f của `report_template.md`:
giới hạn độ sâu, giới hạn số mẫu tại lá, so sánh Gini/Entropy.

## Chạy lại

Từ thư mục gốc dự án, với các thư viện trong `requirements.txt` đã cài:

```powershell
python src/improvements/tree_experiments.py
python -m unittest discover -s tests -v
```

Script lấy đường dẫn theo vị trí file nên cũng chạy được từ thư mục khác.
Không cần chạy `main.py` (file này hiện chứa lẫn Markdown và mã Python).
Không sửa dữ liệu train/test hoặc ghi đè các đầu ra baseline ngày 2.
Chạy lại sẽ ghi đè các đầu ra có tiền tố `improvements_`.

## Đầu ra

- `dataset/improvements_cv_max_depth.csv`, `improvements_cv_min_samples_leaf.csv`,
  `improvements_cv_criterion.csv`: điểm train/validation của từng fold, từng ứng viên.
- `dataset/improvements_results.csv`: tham số được chọn, accuracy CV, các chỉ số
  train/test, độ phức tạp cây và confusion matrix (TN, FP, FN, TP).
- `dataset/improvements_predictions.csv`: nhãn thực tế và dự đoán của bốn mô hình;
  thứ tự hàng giống tập test.
- `model/improvements_*.joblib`: baseline huấn luyện lại và ba mô hình sau lựa chọn.
- `doc/improvements_run.json`: cấu hình, kích thước dữ liệu, phiên bản scikit-learn.
- `img/improvements_comparison.png`: so sánh chất lượng và độ phức tạp cây.
- `doc/report_template.md`, phần f: phương pháp, kết quả và nhận xét tiếng Việt.

Chỉ nạp joblib do bạn tin cậy. Mô hình nhận 73 cột đã mã hóa theo đúng thứ
tự train, không nhận trực tiếp dữ liệu thô.

## Cách đọc kết quả

Accuracy là tiêu chí chọn tham số đã xác định trước; precision/recall/F1
được tính cho lớp mua hàng (1). Các thí nghiệm độc lập, không kết hợp tham số.
`min_samples_leaf=50` đứng đầu CV; `max_depth=5` đứng đầu test quan sát được.
Không chọn lại theo test. Số liệu là một lần chia dữ liệu, không phải khẳng
định mô hình luôn tốt hơn trên dữ liệu mới.

Lần chạy này dùng scikit-learn 1.6.1, pandas 3.0.3, matplotlib 3.11.1;
pandas và matplotlib sẵn có khác phiên bản ghim trong requirements.txt.
Các kiểm tra đi kèm đối chiếu mô hình lưu với dự đoán và chỉ số xuất ra.
