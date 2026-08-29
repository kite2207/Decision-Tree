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

[Phần này sẽ được Người 1 và Người 2 điền sau khi xử lý dữ liệu]

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

### Cải thiện 1: Giới hạn độ sâu cây (max_depth)

[Người 2 - Ngày 3]

- **Mô tả phương pháp:** ...
- **Kết quả:** ...
- **Giải thích:** ...

### Cải thiện 2: Giới hạn số mẫu tối thiểu tại lá (min_samples_leaf)

[Người 2 - Ngày 3]

- **Mô tả phương pháp:** ...
- **Kết quả:** ...
- **Giải thích:** ...

### Cải thiện 3: So sánh tiêu chí phân chia (Gini vs. Entropy) hoặc Pruning

[Người 3 - Ngày 3]

- **Mô tả phương pháp:** ...
- **Kết quả:** ...
- **Giải thích:** ...

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
