# Phan tich Cay Quyet Dinh Baseline

## 1. Tong quan cay

Mo hinh baseline su dung thuat toan `DecisionTreeClassifier` cua scikit-learn voi tieu chi chia nut la **Gini** va khong dat bat ky rang buoc nao (max_depth=None, min_samples_leaf=1). Ket qua:

| Thong so | Gia tri |
|---|---|
| Chieu sau cay (depth) | 26 |
| Tong so nut (nodes) | 1,627 |
| So nut la (leaves) | 814 |
| So nut phan chia (internal) | 813 |
| So dac trung dung trong phan chia | 61 / 73 |

**Nhan xet:** Cay co 26 tang va 1,627 nut la mot cay **rat lon** va **phuc tap**. Muc do phuc tap nay la dau hieu ro rang cua **overfitting** — tren du lieu huan luyen, mo hinh dat accuracy 100% (1.0000), nhung tren du lieu kiem tra chi dat 85.58%. Su chech lech nay cho thay cay da "hoc thuoc" du lieu huan luyen thay vi hoc cac quy luat tong quat.

---

## 2. Dieu kien phan chia dau tien (Root Node)

**Nut goc** cua cay la dieu kien quan trong nhat, vi no phan chia toan bo du lieu ngay tu dau:

```
remainder__PageValues <= 0.9448
```

| Nhanh | Dieu kien | Y nghia |
|---|---|---|
| Trai | PageValues <= 0.9448 | Phien truy cap co chi so trang thap → it co kha nang mua hang |
| Phai | PageValues > 0.9448 | Phien truy cap co chi so trang cao → co kha nang mua hang |

**`PageValues`** la dac trung quan trong nhat trong toan bo mo hinh, voi muc do anh huong (feature importance) len den **42.00%** — cao hon rat nhieu so voi cac dac trung con lai. Day la gia tri trung binh cua cac trang ma nguoi dung truy cap truoc khi thuc hien giao dich. Gia tri nay cao → nguoi dung da xem nhieu trang san pham co gia tri → co xu huong mua hang.

---

## 3. Cac dac trung xuat hien nhieu nhat trong cay

| Hang | Dac trung | Muc do quan trong |
|---|---|---|
| 1 | remainder__PageValues | 42.00% |
| 2 | remainder__ProductRelated_Duration | 8.63% |
| 3 | remainder__BounceRates | 6.81% |
| 4 | remainder__ExitRates | 6.33% |
| 5 | remainder__Administrative_Duration | 4.83% |
| 6 | remainder__Administrative | 4.37% |
| 7 | remainder__ProductRelated | 4.03% |
| 8 | categorical__Month_Nov | 2.48% |
| 9 | remainder__Informational_Duration | 2.27% |
| 10 | remainder__Informational | 1.31% |

Nhin chung, cac dac trung lien quan den **hanh vi duyet trang** (thoi gian, so luong trang da xem, ty le thoat) dong vai tro chu dao, trong khi cac bien phan loai (thang, loai thiet bi, v.v.) co anh huong nho hon.

---

## 4. Phan tich hai duong di tieu bieu

### Duong di 1: Nhanh TRAI tai goc — Du doan "Khong mua hang"

Dieu kien: `PageValues <= 0.9448` (DUNG — gia tri trang thap)

```
Node 0 : PageValues <= 0.9448
  Node 1 : Month_Nov <= 0.5000
    Node 2 : Administrative_Duration <= 1.0000
      Node 3 : PageValues <= 0.0696
        Node 4 : Month_Dec <= 0.5000
          Node 5 : Informational <= 4.5000
            Node 6 : TrafficType_20 <= 0.5000
              Node 7 : Month_Aug <= 0.5000
                -> LEAF: "No Purchase" (n=1)
```

**Giai thich:**
Nguoi dung co gia tri trang thap (`PageValues <= 0.9448`), khong truy cap vao thang 11, danh it thoi gian tren trang hanh chinh (`Administrative_Duration <= 1 giay`), co gia tri trang gan nhu bang 0 (`PageValues <= 0.0696`), khong vao thang 12, xem it trang thong tin (`Informational <= 4`), khong den tu nguon luu luong 20, va khong truy cap vao thang 8. Ket qua: cay du doan nguoi nay **KHONG mua hang**.

Day la hanh vi cua mot nguoi dung "lut lan" qua trang web, khong the hien su quan tam cu the den san pham.

---

### Duong di 2: Nhanh PHAI tai goc — Du doan "Mua hang"

Dieu kien: `PageValues > 0.9448` (SAI — gia tri trang cao)

```
Node 0  : PageValues > 0.9448
  Node 822 : BounceRates <= 0.0004
    Node 823 : PageValues <= 14.7979
      Node 824 : Administrative <= 4.5000
        Node 825 : ProductRelated_Duration <= 3100.0060
          Node 826 : ExitRates <= 0.0408
            Node 827 : VisitorType_Returning_Visitor <= 0.5000
              Node 828 : ProductRelated_Duration <= 2403.7333
                -> LEAF: "Purchase" (n=1)
```

**Giai thich:**
Nguoi dung co gia tri trang cao (`PageValues > 0.9448`), ty le thoat ngay (`BounceRates`) rat thap (khong bo trang), gia tri trang o muc vua phai (`PageValues <= 14.7979`), da xem it trang hanh chinh, thoi gian xem trang san pham vua du (`ProductRelated_Duration <= 3100 giay`), ty le thoat khoi trang thap, khong phai nguoi truy cap quay lai (`VisitorType_Returning != 1`), va thoi gian tren trang san pham trong muc 2403 giay. Ket qua: cay du doan nguoi nay **CO mua hang**.

Day la hanh vi cua mot nguoi dung co muc dich ro rang — ho xem nhieu trang co gia tri, khong bo di ngay, va danh du thoi gian can thiet.

---

## 5. Nhan xet tong the va diem yeu cua cay baseline

### Diem manh
- Dat accuracy **85.58%** tren tap kiem tra — ket qua kha tot cho mo hinh ban dau.
- Xac dinh chinh xac `PageValues` la dac trung quan trong nhat.
- Khong can tinh chinh tham so — mo hinh don gian de xay dung.

### Diem yeu va van de
1. **Overfitting nghiem trong**: Train accuracy = 100% nhung test accuracy = 85.58%. Cay da "hoc thuoc" tap huan luyen.
2. **Cay qua lon**: 26 tang, 1,627 nut, 814 la. Kho doc, kho giai thich.
3. **Ket qua kem voi lop thieu so**: F1-score cua lop "Purchase" chi la **0.5544** — the hien mo hinh du doan sai nhieu voi cac phien co mua hang (lop thieu so: 382/2441 = 15.6%).
4. **Bias ve lop "No Purchase"**: Vi du lieu mat can bang (84.4% vs 15.6%), mo hinh co xu huong du doan "No Purchase" nhieu hon.

### Huong cai thien
- **Gioi han chieu sau** (`max_depth`): Giam overfitting, don gian hoa cay.
- **Tang min_samples_leaf**: Tranh cac la co it mau (tien hanh boi Nguoi 2).
- **Thay doi tieu chi chia** (Gini vs Entropy): Kiem tra xem tieu chi nao phu hop hon (tien hanh boi Nguoi 3 — xem ket qua improvement3_result.txt).

---

*File nay duoc tao tu: `src/person3/analyze_tree.py`*
*Du lieu tham khao: `doc/baseline_result.txt`, `doc/tree_analysis_raw.txt`*
