import csv
from pathlib import Path

import joblib
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "online_shoppers_train.csv"
TEST_FILE = BASE_DIR / "online_shoppers_test.csv"
MODEL_FILE = BASE_DIR / "baseline_model.joblib"
RESULT_FILE = BASE_DIR / "baseline_result.txt"
CLASS_TABLE_FILE = BASE_DIR / "baseline_class_results.csv"
PREDICTION_FILE = BASE_DIR / "baseline_predictions.csv"
CONFUSION_IMAGE_FILE = BASE_DIR / "baseline_confusion_matrix.png"
TARGET = "Revenue"
LABELS, DISPLAY_NAMES = [0, 1], ["No Purchase", "Purchase"]


def load_baseline_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise ValueError(f"Cả hai tập dữ liệu phải có cột {TARGET}.")

    feature_columns = [column for column in train_df.columns if column != TARGET]
    test_feature_columns = [column for column in test_df.columns if column != TARGET]
    if feature_columns != test_feature_columns:
        raise ValueError("Các đặc trưng của tập train và test không khớp nhau.")

    return (
        train_df[feature_columns],
        train_df[TARGET].to_numpy(),
        test_df[feature_columns],
        test_df[TARGET].to_numpy(),
        feature_columns,
    )


def save_class_table(rows):
    with CLASS_TABLE_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["correct_percentage"] = f'{row["correct_percentage"]:.2f}'
            output["wrong_percentage"] = f'{row["wrong_percentage"]:.2f}'
            writer.writerow(output)


def save_predictions(y_true, y_pred):
    with PREDICTION_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual", "predicted", "is_correct"])
        writer.writerows((int(a), int(p), bool(a == p)) for a, p in zip(y_true, y_pred))


def save_confusion_matrix_image(matrix):
    image = Image.new("RGB", (900, 680), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    title_font = ImageFont.load_default(size=32)
    draw.text((130, 35), "Baseline Decision Tree - Confusion Matrix", fill="black", font=title_font)
    left, top, cell = 230, 180, 210
    maximum = int(matrix.max()) or 1
    for row in range(2):
        for column in range(2):
            value = int(matrix[row, column])
            intensity = int(245 - 160 * value / maximum)
            box = (left + column * cell, top + row * cell, left + (column + 1) * cell, top + (row + 1) * cell)
            draw.rectangle(box, fill=(intensity, intensity, 255), outline="navy", width=3)
            text = str(value)
            bbox = draw.textbbox((0, 0), text, font=title_font)
            draw.text((box[0] + (cell - bbox[2] + bbox[0]) / 2, box[1] + (cell - bbox[3] + bbox[1]) / 2), text, fill="black", font=title_font)
    for index, name in enumerate(DISPLAY_NAMES):
        draw.text((left + index * cell + 35, top - 45), name, fill="black", font=font)
        draw.text((15, top + index * cell + 90), name, fill="black", font=font)
    draw.text((360, 625), "Predicted label", fill="black", font=font)
    draw.text((15, 135), "Actual label", fill="black", font=font)
    image.save(CONFUSION_IMAGE_FILE)


def main():
    X_train, y_train, X_test, y_test, feature_columns = load_baseline_data()
    model = joblib.load(MODEL_FILE)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=LABELS)
    report = classification_report(y_test, y_pred, labels=LABELS, target_names=DISPLAY_NAMES, digits=4, zero_division=0)

    rows = []
    for index, (label, name) in enumerate(zip(LABELS, DISPLAY_NAMES)):
        actual_count = int(matrix[index, :].sum())
        correct = int(matrix[index, index])
        wrong = actual_count - correct
        rows.append({"class": label, "label": name, "actual_count": actual_count,
                     "correct": correct, "wrong": wrong,
                     "correct_percentage": correct / actual_count * 100,
                     "wrong_percentage": wrong / actual_count * 100})
    save_class_table(rows)
    save_predictions(y_test, y_pred)
    save_confusion_matrix_image(matrix)
    table_lines = [f'{r["label"]:<12} | tổng {r["actual_count"]:>4} | đúng {r["correct"]:>4} ({r["correct_percentage"]:>6.2f}%) | sai {r["wrong"]:>4} ({r["wrong_percentage"]:>6.2f}%)' for r in rows]
    correct_total = int((y_test == y_pred).sum())
    result = f"""KẾT QUẢ ĐÁNH GIÁ CÂY QUYẾT ĐỊNH BASELINE
============================================================
Dữ liệu huấn luyện: {TRAIN_FILE.name}
Dữ liệu kiểm tra: {TEST_FILE.name}
Mô hình: {MODEL_FILE.name}
Số mẫu huấn luyện: {len(y_train):,}
Số mẫu kiểm tra: {len(y_test):,}
Số đặc trưng: {len(feature_columns)}

KẾT QUẢ TỔNG QUÁT
------------------------------------------------------------
Số dự đoán đúng: {correct_total:,}/{len(y_test):,}
Số dự đoán sai: {len(y_test) - correct_total:,}/{len(y_test):,}
Accuracy (tỷ lệ đúng): {accuracy:.4f} ({accuracy * 100:.2f}%)
Error rate (tỷ lệ sai): {1 - accuracy:.4f} ({(1 - accuracy) * 100:.2f}%)

BẢNG ĐÚNG/SAI THEO TỪNG NHÓM KẾT QUẢ
------------------------------------------------------------
{chr(10).join(table_lines)}

CONFUSION MATRIX
------------------------------------------------------------
Hàng = nhãn thực tế; cột = nhãn dự đoán
{matrix}

CLASSIFICATION REPORT
------------------------------------------------------------
{report}
Ghi chú: Vì dữ liệu mất cân bằng, cần đọc thêm precision, recall và F1-score
của lớp Purchase, không chỉ dựa vào accuracy.
"""
    RESULT_FILE.write_text(result, encoding="utf-8")
    print(result)
    print(f"Đã lưu ảnh: {CONFUSION_IMAGE_FILE.name}")


if __name__ == "__main__":
    main()
