# [GroupID] – Decision Tree Report
## Lab 2: Decision Tree Modeling and Improvement
### Introduction to Artificial Intelligence

---

## a. Giới thiệu nhóm (Group Introduction)

| Thông tin | Chi tiết |
|---|---|
| **Tên nhóm** | [Tên nhóm] |
| **Mã nhóm (GroupID)** | [GroupID] |

| STT | Họ và tên | MSSV | Phần đóng góp |
|---|---|---|---|
| 1 | [Tên] | [MSSV] | Tìm và mô tả dataset; tiền xử lý; xây cây baseline |
| 2 | [Tên] | [MSSV] | Đánh giá kết quả; thực hiện Cải thiện 1 và 2 |
| 3 | [Tên] | [MSSV] | Nghiên cứu lý thuyết; thực hiện Cải thiện 3; tổng hợp báo cáo |

---

## b. Giới thiệu (Introduction)

### Cây quyết định là gì?

Cây quyết định (Decision Tree) là một mô hình học máy có giám sát (supervised learning) dùng để giải quyết các bài toán phân loại (classification) và hồi quy (regression). Mô hình có cấu trúc dạng cây với:

- **Nút gốc (Root node):** Điều kiện phân chia đầu tiên, chọn đặc trưng quan trọng nhất.
- **Nút trong (Internal node):** Điều kiện kiểm tra một đặc trưng.
- **Nhánh (Branch):** Kết quả của điều kiện (đúng/sai, lớn hơn/nhỏ hơn, v.v.)
- **Nút lá (Leaf node):** Kết quả dự đoán cuối cùng (nhãn lớp hoặc giá trị).

### Cây quyết định hoạt động như thế nào?

Quá trình xây dựng cây quyết định sử dụng thuật toán phân chia đệ quy (recursive partitioning):

1. Bắt đầu từ toàn bộ tập dữ liệu tại nút gốc.
2. Tại mỗi nút, chọn đặc trưng và ngưỡng phân chia sao cho tạo ra các nhóm con thuần nhất nhất có thể (dựa trên Gini Impurity hoặc Information Gain/Entropy).
3. Lặp lại đệ quy cho đến khi đạt điều kiện dừng.
4. Khi dự đoán, một mẫu dữ liệu mới đi từ nút gốc xuống theo các điều kiện cho đến khi đến nút lá.

**Các tiêu chí phân chia phổ biến:**

| Tiêu chí | Ý nghĩa |
|---|---|
| **Gini Impurity** | Đo mức độ lẫn lộn của một nút; thấp hơn = thuần hơn |
| **Entropy / Information Gain** | Đo lượng thông tin thu được sau khi phân chia |

### Tại sao bộ dữ liệu đã chọn phù hợp với cây quyết định?

Dataset **Online Shoppers Purchasing Intention** phù hợp với cây quyết định vì:

1. **Bài toán phân loại nhị phân rõ ràng:** Biến mục tiêu `Revenue` chỉ có hai giá trị True/False.
2. **Đặc trưng hỗn hợp:** Cả numerical lẫn categorical, cây quyết định xử lý được cả hai loại.
3. **Không cần chuẩn hóa dữ liệu:** Cây quyết định không yêu cầu feature scaling.
4. **Dễ diễn giải kết quả:** Cây tạo ra quy tắc if-else rõ ràng, rất có giá trị trong thương mại điện tử.
5. **Kích thước phù hợp:** 12,330 mẫu và 17 đặc trưng đủ lớn để học các mẫu.

### Mục tiêu của dự án

Xây dựng mô hình cây quyết định để dự đoán khách hàng có thực sự mua hàng trong một phiên truy cập hay không, và đề xuất các phương pháp cải thiện hiệu năng.

---

## c. Mô tả dữ liệu (Dataset Description)

### Nguồn dữ liệu

- **Tên dataset:** Online Shoppers Purchasing Intention Dataset
- **Nguồn:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

### Thông tin tổng quan

| Thông tin | Giá trị |
|---|---|
| Số mẫu (dòng) | 12,330 |
| Số đặc trưng (cột) | 17 |
| Biến mục tiêu | Revenue (True/False) |
| Giá trị bị thiếu | Không có |
| Loại bài toán | Phân loại nhị phân |

### Mô tả các đặc trưng

| Tên cột | Kiểu dữ liệu | Mô tả |
|---|---|---|
| Administrative | int | Số trang thuộc nhóm quản trị đã xem |
| Administrative_Duration | float | Tổng thời gian (giây) trên trang quản trị |
| Informational | int | Số trang thuộc nhóm thông tin đã xem |
| Informational_Duration | float | Tổng thời gian (giây) trên trang thông tin |
| ProductRelated | int | Số trang sản phẩm đã xem |
| ProductRelated_Duration | float | Tổng thời gian (giây) trên trang sản phẩm |
| BounceRates | float | Tỉ lệ thoát trang ngay sau khi vào |
| ExitRates | float | Tỉ lệ thoát trang từ trang đó |
| PageValues | float | Giá trị trung bình của trang trước giao dịch |
| SpecialDay | float | Độ gần so với ngày lễ (0-1) |
| Month | str | Tháng trong năm (Jan, Feb, ...) |
| OperatingSystems | int | Hệ điều hành của khách hàng |
| Browser | int | Trình duyệt của khách hàng |
| Region | int | Vùng địa lý của khách hàng |
| TrafficType | int | Loại nguồn truy cập |
| VisitorType | str | Loại khách (Returning_Visitor, New_Visitor, Other) |
| Weekend | bool | Phiên truy cập vào cuối tuần? |
| **Revenue** | **bool** | **[TARGET] Có thực hiện giao dịch không?** |

### Các bước tiền xử lý 
*chạy file ```pre.py```*

### 1. Kiểm tra dữ liệu

Bộ dữ liệu ban đầu gồm **12.330 mẫu với 18 thuộc tính**. Trong đó có **17 thuộc tính đầu vào** được sử dụng để dự đoán và một thuộc tính mục tiêu là `Revenue`.

Trước tiên, nhóm kiểm tra các giá trị bị thiếu trong toàn bộ bộ dữ liệu. Kết quả cho thấy **không có giá trị bị thiếu** ở bất kỳ thuộc tính nào.

### 2. Xử lý dữ liệu trùng lặp

Tiếp theo, nhóm kiểm tra các bản ghi bị trùng lặp hoàn toàn. Kết quả phát hiện **125 bản ghi trùng lặp**.

Các bản ghi trùng lặp được loại bỏ nhằm tránh việc cùng một quan sát xuất hiện nhiều lần và tạo ảnh hưởng không cần thiết đến quá trình huấn luyện mô hình.

Sau khi loại bỏ 125 bản ghi trùng lặp, bộ dữ liệu còn lại **12.205 mẫu**.

| Nội dung | Số lượng |
|---|---:|
| Số mẫu ban đầu | 12.330 |
| Số bản ghi trùng lặp | 125 |
| Số mẫu sau khi loại bỏ trùng lặp | 12.205 |
| Giá trị thiếu | 0 |

### 3. Xác định biến đầu vào và biến mục tiêu

Trong bài toán này, biến mục tiêu được chọn là `Revenue`. Biến này biểu thị việc một phiên truy cập website có phát sinh giao dịch mua hàng hay không.

Giá trị của `Revenue` được chuyển đổi thành dạng nhị phân:

- `0`: Không phát sinh giao dịch (`False`)
- `1`: Có phát sinh giao dịch (`True`)

17 thuộc tính còn lại được sử dụng làm các biến đầu vào cho mô hình Decision Tree.

Như vậy, bài toán được xác định là **bài toán phân loại nhị phân (Binary Classification)**.

### 4. Phân loại các thuộc tính

Các thuộc tính đầu vào được chia thành ba nhóm dựa trên ý nghĩa và kiểu dữ liệu.

### Các thuộc tính số (Numerical Features)

- `Administrative`
- `Administrative_Duration`
- `Informational`
- `Informational_Duration`
- `ProductRelated`
- `ProductRelated_Duration`
- `BounceRates`
- `ExitRates`
- `PageValues`
- `SpecialDay`

Có tổng cộng **10 thuộc tính số**.

### Các thuộc tính phân loại (Categorical Features)

- `Month`
- `OperatingSystems`
- `Browser`
- `Region`
- `TrafficType`
- `VisitorType`

Có tổng cộng **6 thuộc tính phân loại**.

### Thuộc tính nhị phân (Binary Feature)

- `Weekend`

Mặc dù các thuộc tính `OperatingSystems`, `Browser`, `Region` và `TrafficType` được biểu diễn bằng các giá trị số trong bộ dữ liệu ban đầu, các giá trị này thực chất là mã đại diện cho các nhóm khác nhau. Do đó, chúng được xem là thuộc tính phân loại thay vì thuộc tính số liên tục.

### 5. Phân tích phân bố biến mục tiêu

Nhóm kiểm tra phân bố của biến `Revenue` để xác định mức độ cân bằng giữa hai lớp.

| `Revenue` | Ý nghĩa | Số mẫu | Tỷ lệ |
|---|---|---:|---:|
| 0 | Không mua hàng | 10.297 | 84,37% |
| 1 | Có mua hàng | 1.908 | 15,63% |
| **Tổng** | | **12.205** | **100%** |

Kết quả cho thấy bộ dữ liệu có **mất cân bằng lớp (Class Imbalance)**. Số phiên truy cập không phát sinh giao dịch chiếm 84,37%, trong khi các phiên có phát sinh giao dịch chỉ chiếm 15,63%.

Điều này cần được lưu ý khi đánh giá mô hình. Nếu chỉ sử dụng Accuracy, mô hình có thể đạt độ chính xác tương đối cao ngay cả khi khả năng nhận diện lớp `Purchase` không tốt. Vì vậy, ngoài Accuracy, nhóm sẽ sử dụng thêm **Precision, Recall, F1-score và Confusion Matrix** để đánh giá mô hình.

### 6. Chia tập huấn luyện và tập kiểm tra

Sau khi loại bỏ các bản ghi trùng lặp và xác định biến đầu vào và biến mục tiêu, dữ liệu được chia thành hai tập:

- **80% dữ liệu cho tập huấn luyện**
- **20% dữ liệu cho tập kiểm tra**

Kết quả:

| Tập dữ liệu | Số mẫu |
|---|---:|
| Tập huấn luyện | 9.764 |
| Tập kiểm tra | 2.441 |
| **Tổng** | **12.205** |

Nhóm sử dụng `random_state = 42` để đảm bảo quá trình chia dữ liệu có thể tái lập trong các lần chạy khác nhau.

Do dữ liệu bị mất cân bằng lớp, phương pháp **stratified splitting** được sử dụng dựa trên biến `Revenue`. Phương pháp này giúp duy trì tỷ lệ giữa hai lớp trong cả tập huấn luyện và tập kiểm tra.

Kết quả phân bố:

| Tập | Không mua | Có mua |
|---|---:|---:|
| Training | 84,37% | 15,63% |
| Testing | 84,35% | 15,65% |

Có thể thấy phân bố của hai lớp trong tập huấn luyện và tập kiểm tra gần như tương đương với phân bố của toàn bộ dữ liệu.

### 7. Mã hóa các thuộc tính phân loại

Mô hình Decision Tree được sử dụng trong thư viện Scikit-learn yêu cầu dữ liệu đầu vào ở dạng số. Do đó, sáu thuộc tính phân loại được chuyển đổi bằng phương pháp **One-Hot Encoding**.

Ví dụ, thuộc tính `VisitorType` có thể nhận các giá trị:

- `New_Visitor`
- `Returning_Visitor`
- `Other`

Sau khi mã hóa, các giá trị này được biểu diễn bằng các thuộc tính nhị phân riêng biệt.

Tương tự, One-Hot Encoding được áp dụng cho:

- `Month`
- `OperatingSystems`
- `Browser`
- `Region`
- `TrafficType`
- `VisitorType`

Phương pháp này được sử dụng thay vì gán trực tiếp các giá trị số cho từng category, vì các category không có quan hệ thứ tự tự nhiên.

Bộ dữ liệu ban đầu có **17 thuộc tính đầu vào**. Sau khi thực hiện One-Hot Encoding, số lượng thuộc tính đầu vào tăng lên **73 thuộc tính**. Sự gia tăng này là do một thuộc tính phân loại được tách thành nhiều thuộc tính nhị phân tương ứng với các giá trị khác nhau của nó.
### 8. Kết quả sau tiền xử lý

Sau khi hoàn thành quá trình tiền xử lý, dữ liệu có các đặc điểm sau:

| Thành phần | Kết quả |
|---|---:|
| Số mẫu ban đầu | 12.330 |
| Bản ghi trùng lặp đã loại bỏ | 125 |
| Số mẫu cuối cùng | 12.205 |
| Số thuộc tính đầu vào ban đầu | 17 |
| Số thuộc tính sau One-Hot Encoding | 73 |
| Tập huấn luyện | 9.764 |
| Tập kiểm tra | 2.441 |
| Giá trị thiếu | 0 |
| Chuẩn hóa | Không thực hiện |
| Mã hóa biến phân loại | One-Hot Encoding |
| Mã hóa `Revenue` | 0 = Không mua, 1 = Có mua |
| Tỷ lệ chia dữ liệu | 80% / 20% |
| Stratified Split | Có |

### Quy trình tiền xử lý

```text
Dữ liệu ban đầu
12.330 mẫu × 18 thuộc tính
          │
          ▼
    Kiểm tra dữ liệu
          │
          ▼
    Không có giá trị thiếu
          │
          ▼
   Loại bỏ 125 bản ghi trùng
          │
          ▼
    12.205 mẫu
          │
          ▼
   Xác định X và y
          │
          ├── X: 17 thuộc tính
          │
          └── y: Revenue
                  0 = Không mua
                  1 = Có mua
          │
          ▼
   Chia dữ liệu 80/20
          │
       ┌──┴──┐
       ▼     ▼
    Train   Test
    9.764   2.441
       │     │
       ▼     ▼
   One-Hot Encoding
          │
          ▼
   73 thuộc tính đầu vào
          │
          ▼
   Sẵn sàng xây dựng
   Decision Tree
```

Sau bước tiền xử lý, dữ liệu đã ở dạng phù hợp để tiến hành xây dựng **mô hình Decision Tree cơ sở (Baseline Decision Tree)** và thực hiện các thí nghiệm cải thiện mô hình ở các bước tiếp theo.

---

## d. Mô hình cây ban đầu (Baseline Decision Tree Model)

[Phần này sẽ được Người 1 thực hiện vào Ngày 2]

- Mô tả cài đặt mô hình ban đầu.
- Quy trình huấn luyện và kiểm tra.
- Hình ảnh cây quyết định.
- Kết quả accuracy và error rate của mô hình baseline.

---

## e. Phân tích cây quyết định (Analysis of the Tree)

[Phần này sẽ được Người 3 thực hiện vào Ngày 2]

- Hình ảnh cây được tạo ra.
- Nhận xét về cấu trúc cây và các nút quyết định quan trọng.
- Điểm mạnh và điểm yếu của cây ban đầu.

---

## f. Các phương pháp cải thiện (Improvement Methods)

### Thiết lập thí nghiệm cải thiện cây

Sử dụng nguyên tập train (9.764 mẫu), test (2.441 mẫu) và 73 đặc trưng đã
tiền xử lý. Mỗi thí nghiệm chỉ thay đổi một tham số, các tham số khác giữ
như baseline (`criterion="gini"`, `random_state=42`). Dùng StratifiedKFold
5 folds, shuffle=True, random_state=42 trên train; chọn cấu hình theo accuracy
trung bình validation. Sau đó huấn luyện lại trên toàn bộ train và đánh giá test.
Không dùng test để chọn tham số. Precision, recall, F1 bên dưới dành cho lớp
Purchase (`Revenue=1`), nhằm bổ sung góc nhìn cho dữ liệu mất cân bằng.

Baseline được huấn luyện lại với cấu hình gốc, cho accuracy test 85,58%,
khớp kết quả ngày 2; không ghi đè mô hình baseline cũ.

### Cải thiện 1: Giới hạn độ sâu cây (max_depth)

- **Mô tả phương pháp:** Thử `max_depth ∈ {3, 5, 7, 10, 15, 20, None}`.
  Hạn chế độ sâu để giảm các nhánh quá chi tiết. CV chọn `max_depth=5`,
  accuracy validation trung bình 89,86%.
- **Kết quả:** Accuracy train 90,99%, test 90,21%; error rate 9,79%;
  precision 70,61%, recall 64,14%, F1 67,22%. Cây sâu 5, có 32 lá và 63 nút.
- **Giải thích:** Accuracy test tăng 4,63 điểm phần trăm so với baseline,
  trong khi chênh lệch train–test giảm từ 14,42 xuống 0,78 điểm phần trăm.
  Kết quả phù hợp với việc giảm overfitting; cây cũng dễ diễn giải hơn.

### Cải thiện 2: Giới hạn số mẫu tối thiểu tại lá (min_samples_leaf)

- **Mô tả phương pháp:** Thử `min_samples_leaf ∈ {1, 2, 5, 10, 20, 50, 100}`.
  Mỗi lá phải chứa ít nhất số mẫu chỉ định, hạn chế quyết định dựa trên
  những nhóm quá nhỏ. CV chọn `min_samples_leaf=50`, accuracy trung bình 90,15%.
- **Kết quả:** Accuracy train 90,96%, test 90,00%; error rate 10,00%;
  precision 71,97%, recall 59,16%, F1 64,94%. Cây sâu 14, có 90 lá và 179 nút.
- **Giải thích:** Accuracy test tăng 4,42 điểm phần trăm so với baseline.
  So với giới hạn độ sâu, precision cao hơn nhưng recall thấp hơn: mô hình
  báo nhầm mua hàng ít hơn (88 so với 102), nhưng bỏ sót nhiều hơn (156 so với 137).

### Cải thiện 3: So sánh tiêu chí phân chia Gini và Entropy

- **Mô tả phương pháp:** So sánh `criterion="gini"` và `criterion="entropy"`
  với cây không giới hạn độ sâu, `min_samples_leaf=1`. Chỉ thay tiêu chí
  phân chia để tách biệt ảnh hưởng với hai thí nghiệm trên; không thực hiện
  thêm pruning vì kế hoạch cho phép chọn một trong hai phương án.
- **Kết quả:** Accuracy CV của Gini là 85,94%, Entropy là 86,53%, nên chọn
  Entropy. Accuracy train 100%, test 86,15%; error rate 13,85%; precision
  55,39%, recall 59,16%, F1 57,22%. Cây sâu 30, có 763 lá và 1.525 nút
  (Gini: sâu 26, 814 lá, 1.627 nút).
- **Giải thích:** Entropy tăng accuracy test 0,57 điểm phần trăm nhưng
  chênh lệch train–test vẫn lớn (13,85 điểm phần trăm). Đổi tiêu chí phân
  chia riêng lẻ chưa khắc phục tốt overfitting trong thí nghiệm này.

### Tổng hợp thí nghiệm cải thiện cây

| Cấu hình | Accuracy CV | Accuracy test | Error rate | F1 Purchase | Độ sâu | Số lá |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Gini | 85,94% | 85,58% | 14,42% | 55,44% | 26 | 814 |
| max_depth=5 | 89,86% | 90,21% | 9,79% | 67,22% | 5 | 32 |
| min_samples_leaf=50 | 90,15% | 90,00% | 10,00% | 64,94% | 14 | 90 |
| Entropy | 86,53% | 86,15% | 13,85% | 57,22% | 30 | 763 |

Độ sâu được tính theo số cạnh từ gốc tới lá sâu nhất (`get_depth()`), không
phải số tầng; số tầng bằng độ sâu + 1.

Theo tiêu chí chọn đã định trước, `min_samples_leaf=50` là cấu hình được
chọn bằng CV. `max_depth=5` có accuracy và F1 test cao nhất trong lần đánh giá
này, nhưng không dùng kết quả test để thay đổi lựa chọn. Chênh lệch nhỏ giữa
hai cấu hình không chứng minh ưu thế có ý nghĩa thống kê. Chưa thử kết hợp
các tham số. Encoder của dữ liệu có sẵn được fit trên toàn bộ train trước
CV; CV ở đây đánh giá mô hình trên đặc trưng đã mã hóa, không phải toàn bộ
pipeline tiền xử lý. Test không tham gia fit encoder.

Mã chạy: `python src/improvements/tree_experiments.py` từ thư mục dự án.
Chi tiết tái lập và danh sách đầu ra nằm trong `doc/improvements_readme.md`.
Phần g và h vẫn dành cho tổng hợp ngày 4.

---

## g. So sánh kết quả (Comparison of Results)

[Người 2 và 3 - Ngày 4]

| Mô hình | Tỷ lệ đúng (Accuracy) | Tỷ lệ sai (Error Rate) | Số tầng cây | Nhận xét |
|---|---|---|---|---|
| Cây ban đầu (Baseline) | ... | ... | ... | ... |
| Cải thiện 1 (max_depth) | ... | ... | ... | ... |
| Cải thiện 2 (min_samples_leaf) | ... | ... | ... | ... |
| Cải thiện 3 (Gini vs Entropy / Pruning) | ... | ... | ... | ... |

---

## h. Kết luận (Conclusion)

[Người 3 - Ngày 4]

- Nhóm đã làm được gì?
- Cách nào cho kết quả tốt nhất?
- Nhóm học được gì từ dự án này?
- Nhận xét về hiệu quả của cây quyết định với bộ dữ liệu này.

---

## i. Tài liệu tham khảo (References)

1. C. Sakar et al., "Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks," Neural Computing and Applications, 2019. https://doi.org/10.1007/s00521-018-3523-0

2. Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O'Reilly Media. (Chapter 6: Decision Trees)

3. Scikit-learn Developers. (2024). Decision Trees - scikit-learn documentation. https://scikit-learn.org/stable/modules/tree.html

4. UCI Machine Learning Repository. Online Shoppers Purchasing Intention Dataset. https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

5. Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984). Classification and Regression Trees. CRC Press.
