# **Kế hoạch chi tiết trong 5 ngày**

## **Ngày 1: Chọn và tìm hiểu dữ liệu**

### **Người 1: Tìm bộ dữ liệu**

Cần làm:

1. Tìm một bộ dữ liệu từ Kaggle, UCI hoặc OpenML.  
2. Tải dữ liệu về dưới dạng `.csv`.  
3. Xác định:  
   * Dữ liệu dùng để dự đoán điều gì?  
   * Cột nào là kết quả cần dự đoán?  
   * Các cột còn lại có ý nghĩa gì?  
   * Có bao nhiêu dòng và bao nhiêu cột?  
4. Kiểm tra dữ liệu có ô bị trống hay không.  
5. Gửi bộ dữ liệu cho cả nhóm.

Sản phẩm cuối ngày:

* File dữ liệu, ví dụ: `dataset.csv`.  
* File ghi thông tin dữ liệu: `dataset_information.txt`.  
* Đường dẫn đến trang đã tải dữ liệu.

---

### **Người 2: Kiểm tra bộ dữ liệu**

Sau khi nhận dữ liệu từ Người 1, cần làm:

1. Mở file dữ liệu và kiểm tra từng cột.  
2. Xác định cột nào chứa số, cột nào chứa chữ.  
3. Kiểm tra có dòng nào bị trùng lặp không.  
4. Kiểm tra số lượng của từng nhóm kết quả.  
5. Ghi lại những vấn đề cần xử lý, ví dụ:  
   * Có ô trống.  
   * Có dữ liệu viết bằng chữ.  
   * Một nhóm có quá nhiều dữ liệu so với nhóm khác.

Sản phẩm cuối ngày:

* File `data_check.txt` ghi rõ các vấn đề của dữ liệu.  
* Bảng thống kê số lượng của từng nhóm kết quả.

---

### **Người 3: Tìm hiểu yêu cầu và chuẩn bị nội dung giải thích**

Cần làm:

1. Đọc toàn bộ yêu cầu của bài Lab.  
2. Viết ngắn gọn:  
   * Cây quyết định là gì?  
   * Cây hoạt động như thế nào?  
   * Tại sao bộ dữ liệu đã chọn phù hợp với cây quyết định?  
3. Tìm 2–3 tài liệu tham khảo đáng tin cậy.  
4. Tạo sẵn khung báo cáo với các phần:  
   * Giới thiệu nhóm.  
   * Giới thiệu bài toán.  
   * Mô tả dữ liệu.  
   * Cây ban đầu.  
   * Các cách cải thiện.  
   * So sánh kết quả.  
   * Kết luận.  
   * Tài liệu tham khảo.

Sản phẩm cuối ngày:

* File khung báo cáo.  
* Phần giới thiệu về cây quyết định.  
* Danh sách tài liệu tham khảo.

---

## **Ngày 2: Xây dựng cây quyết định ban đầu**

### **Người 1: Viết chương trình chính**

Cần làm:

1. Viết code đọc file dữ liệu.  
2. Xử lý các ô trống nếu có.  
3. Chuyển những dữ liệu dạng chữ sang dạng mà chương trình có thể sử dụng.  
4. Chia dữ liệu thành hai phần:  
   * Một phần để chương trình học.  
   * Một phần để kiểm tra kết quả.  
5. Tạo cây quyết định đầu tiên.  
6. Chạy chương trình và lưu kết quả dự đoán.  
7. Ghi rõ các thiết lập đã sử dụng.

Sản phẩm cuối ngày:

* File code tạo cây ban đầu, ví dụ `baseline_model.py`.  
* File hoặc ảnh kết quả chạy chương trình.  
* Bảng ghi kết quả của cây ban đầu.

---

### **Người 2: Viết phần đánh giá kết quả**

Cần làm:

1. Nhận kết quả dự đoán từ Người 1\.  
2. Tính:  
   * Bao nhiêu phần trăm dự đoán đúng.  
   * Bao nhiêu phần trăm dự đoán sai.  
   * Mỗi nhóm được dự đoán đúng và sai bao nhiêu lần.  
3. Tạo bảng thể hiện số lần dự đoán đúng và sai của từng nhóm.  
4. Lưu kết quả để sau này so sánh với các cây đã cải thiện.

Sản phẩm cuối ngày:

* File code đánh giá kết quả, ví dụ `evaluation.py`.  
* Ảnh bảng dự đoán đúng và sai.  
* File `baseline_result.txt` ghi toàn bộ kết quả ban đầu.

---

### **Người 3: Tạo hình và phân tích cây ban đầu**

Cần làm:

1. Nhận cây quyết định từ Người 1\.  
2. Xuất cây thành hình ảnh.  
3. Quan sát và ghi lại:  
   * Cây có bao nhiêu tầng?  
   * Điều kiện đầu tiên của cây là gì?  
   * Những cột dữ liệu nào xuất hiện nhiều trong cây?  
   * Cây có quá lớn và khó đọc không?  
4. Chọn 2–3 đường đi trong cây và giải thích:  
   * Nếu dữ liệu thỏa điều kiện nào thì đi sang trái?  
   * Nếu không thỏa thì đi sang phải?  
   * Cuối cùng cây đưa ra kết quả gì?

Sản phẩm cuối ngày:

* Ảnh `baseline_tree.png`.  
* Phần phân tích cây ban đầu khoảng 1–2 trang.  
* Giải thích ít nhất hai đường đi trong cây.

---

## **Ngày 3: Thử các cách cải thiện cây**

### **Người 1: Hoàn thiện cây ban đầu và hỗ trợ tích hợp**

Cần làm:

1. Kiểm tra code của ngày 2 có chạy ổn định không.  
2. Sắp xếp code thành các phần rõ ràng:  
   * Đọc dữ liệu.  
   * Chuẩn bị dữ liệu.  
   * Tạo cây.  
   * Đánh giá kết quả.  
   * Xuất hình cây.  
3. Đảm bảo cả nhóm sử dụng cùng một cách chia dữ liệu.  
4. Hỗ trợ Người 2 và Người 3 nếu code cải thiện không chạy được.  
5. Tạo một file chung để chạy tất cả các mô hình.

Sản phẩm cuối ngày:

* File `main.py` có thể chạy cây ban đầu và các cây cải thiện.  
* Thư mục code được sắp xếp rõ ràng.  
* File hướng dẫn cách chạy chương trình.

---

### **Người 2: Thực hiện cách cải thiện thứ nhất và thứ hai**

#### **Cách 1: Giới hạn số tầng của cây**

Cần làm:

1. Chạy cây với nhiều số tầng khác nhau, ví dụ:  
   * 3 tầng.  
   * 5 tầng.  
   * 7 tầng.  
   * 10 tầng.  
2. Ghi lại kết quả đúng và sai của mỗi lần.  
3. Xác định số tầng nào cho kết quả tốt nhất.  
4. Giải thích:  
   * Cây quá ít tầng có vấn đề gì?  
   * Cây quá nhiều tầng có vấn đề gì?  
   * Vì sao chọn số tầng tốt nhất?

#### **Cách 2: Giới hạn số dữ liệu ở cuối mỗi nhánh**

Cần làm:

1. Thử yêu cầu mỗi điểm cuối của cây phải có ít nhất:  
   * 1 dòng dữ liệu.  
   * 2 dòng dữ liệu.  
   * 5 dòng dữ liệu.  
   * 10 dòng dữ liệu.  
2. Ghi kết quả của mỗi lần thử.  
3. Chọn giá trị tốt nhất.  
4. So sánh với cây ban đầu.

Sản phẩm cuối ngày:

* Code của hai cách cải thiện.  
* Bảng kết quả của tất cả lần thử.  
* Hai đoạn giải thích vì sao kết quả tăng hoặc giảm.  
* Ảnh cây tốt nhất của mỗi cách.

---

### **Người 3: Thực hiện cách cải thiện thứ ba**

Có thể chọn một trong hai cách sau:

#### **Lựa chọn A: Thử hai cách chia nhánh khác nhau**

1. Chạy cây bằng cách chia nhánh `Gini`.  
2. Chạy cây bằng cách chia nhánh `Entropy`.  
3. So sánh:  
   * Tỷ lệ dự đoán đúng.  
   * Tỷ lệ dự đoán sai.  
   * Kích thước của cây.  
4. Kết luận cách nào phù hợp hơn.

#### **Lựa chọn B: Cắt bớt những nhánh không cần thiết**

1. Tạo cây đầy đủ.  
2. Thử cắt bớt các nhánh nhỏ hoặc ít quan trọng.  
3. Chạy chương trình sau mỗi lần cắt.  
4. Tìm cây vừa nhỏ gọn vừa giữ được kết quả tốt.  
5. Xuất hình cây trước và sau khi cắt.

Sản phẩm cuối ngày:

* Code của cách cải thiện thứ ba.  
* Bảng kết quả trước và sau khi cải thiện.  
* Ảnh cây đã cải thiện.  
* Phần giải thích kết quả.

---

## **Ngày 4: So sánh kết quả và viết báo cáo**

### **Người 1: Ghép và kiểm tra code**

Cần làm:

1. Nhận code của Người 2 và Người 3\.  
2. Ghép vào chương trình chính.  
3. Chạy lần lượt:  
   * Cây ban đầu.  
   * Cách cải thiện 1\.  
   * Cách cải thiện 2\.  
   * Cách cải thiện 3\.  
4. Kiểm tra tất cả đều sử dụng cùng bộ dữ liệu.  
5. Kiểm tra kết quả có được in và lưu đầy đủ không.  
6. Viết file hướng dẫn chạy code.

Sản phẩm cuối ngày:

* Thư mục code hoàn chỉnh.  
* File `README.txt` hướng dẫn cài đặt và chạy.  
* Toàn bộ hình ảnh và kết quả được lưu đúng thư mục.

---

### **Người 2: Tạo bảng so sánh**

Cần tạo bảng theo mẫu:

| Mô hình | Tỷ lệ đúng | Tỷ lệ sai | Số tầng của cây | Nhận xét |
| ----- | ----- | ----- | ----- | ----- |
| Cây ban đầu | ... | ... | ... | Cây lớn hoặc nhỏ |
| Cải thiện 1 | ... | ... | ... | ... |
| Cải thiện 2 | ... | ... | ... | ... |
| Cải thiện 3 | ... | ... | ... | ... |

Sau đó cần:

1. Chỉ ra mô hình có kết quả tốt nhất.  
2. Chỉ ra mô hình có cây dễ hiểu nhất.  
3. Giải thích vì sao mô hình tốt nhất tốt hơn cây ban đầu.  
4. Viết phần báo cáo về các cách cải thiện.

Sản phẩm cuối ngày:

* Bảng so sánh hoàn chỉnh.  
* Biểu đồ so sánh tỷ lệ đúng.  
* Phần báo cáo về các cách cải thiện.

---

### **Người 3: Tổng hợp báo cáo**

Cần làm:

1. Nhận nội dung từ Người 1 và Người 2\.  
2. Đưa nội dung vào đúng phần của báo cáo.  
3. Thêm:  
   * Hình cây ban đầu.  
   * Hình cây sau khi cải thiện.  
   * Bảng kết quả.  
   * Biểu đồ so sánh.  
4. Viết phần phân tích chung.  
5. Viết kết luận:  
   * Nhóm đã làm được gì?  
   * Cách nào tốt nhất?  
   * Nhóm học được gì?  
6. Kiểm tra tài liệu tham khảo.

Sản phẩm cuối ngày:

* Bản báo cáo gần hoàn chỉnh.  
* Nội dung trình bày thống nhất.  
* Tất cả hình và bảng đều có tên, số thứ tự và lời giải thích.

---

## **Ngày 5: Kiểm tra, quay video và nộp bài**

### **Người 1: Kiểm tra chương trình và đóng gói**

Cần làm:

1. Chạy code trên máy khác hoặc một thư mục mới.  
2. Kiểm tra có thiếu file dữ liệu hoặc thư viện không.  
3. Sửa các đường dẫn bị sai.  
4. Kiểm tra file hướng dẫn chạy.  
5. Chuẩn bị thư mục nộp bài theo đúng tên được yêu cầu.  
6. Sau khi video và báo cáo hoàn thành, đóng gói thành một file ZIP.

---

### **Người 2: Chuẩn bị và quay phần cải thiện**

Cần trình bày:

1. Kết quả của cây ban đầu.  
2. Hai cách cải thiện đã thử.  
3. Các giá trị đã thử trong mỗi cách.  
4. Bảng kết quả.  
5. Cách nào tốt nhất và vì sao.

Thời lượng đề xuất: **3–4 phút**.

Ngoài ra, Người 2 kiểm tra lại toàn bộ con số trong báo cáo có khớp với kết quả code hay không.

---

### **Người 3: Hoàn thiện báo cáo và video**

Cần làm:

1. Sửa lỗi chính tả và định dạng báo cáo.  
2. Kiểm tra báo cáo có đủ:  
   * Thông tin thành viên.  
   * Mô tả dữ liệu.  
   * Cây ban đầu.  
   * Phân tích cây.  
   * Ba cách cải thiện.  
   * Tỷ lệ đúng và sai.  
   * Bảng so sánh.  
   * Kết luận.  
   * Tài liệu tham khảo.  
3. Xuất báo cáo thành PDF.  
4. Ghép các phần video.  
5. Kiểm tra video có tiếng, có hình và đọc được các bảng.  
6. Gửi file hoàn chỉnh cho Người 1 đóng gói.

