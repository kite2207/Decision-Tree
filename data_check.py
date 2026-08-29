from pathlib import Path

import pandas as pd


INPUT_FILE = Path(__file__).with_name("online_shoppers_intention.csv")
REPORT_FILE = Path(__file__).with_name("data_check.txt")
TARGET_TABLE_FILE = Path(__file__).with_name("target_distribution.csv")
TARGET = "Revenue"


def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    duplicate_count = int(df.duplicated().sum())
    missing_by_column = df.isna().sum()

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    boolean_columns = df.select_dtypes(include="bool").columns.tolist()

    target_table = (
        df[TARGET]
        .value_counts(dropna=False)
        .rename_axis(TARGET)
        .reset_index(name="count")
    )
    target_table["percentage"] = target_table["count"] / len(df) * 100
    target_table["label"] = target_table[TARGET].map(
        {False: "No Purchase", True: "Purchase"}
    ).fillna("Missing")
    target_table = target_table[[TARGET, "label", "count", "percentage"]]
    target_table.to_csv(TARGET_TABLE_FILE, index=False, float_format="%.2f")

    categorical_codes = [
        column
        for column in ["OperatingSystems", "Browser", "Region", "TrafficType"]
        if column in df.columns
    ]
    imbalance_ratio = target_table["count"].max() / target_table["count"].min()

    lines = [
        f"Tệp dữ liệu: {INPUT_FILE.name}",
        f"Số dòng: {len(df):,}",
        f"Số cột: {df.shape[1]}",
        f"Cột mục tiêu: {TARGET}",
        "",
        "1. KIỂU DỮ LIỆU TỪNG CỘT",
        "-" * 50,
    ]
    lines.extend(f"{column}: {dtype}" for column, dtype in df.dtypes.items())
    lines.extend(
        [
            "",
            "2. PHÂN NHÓM CỘT",
            "-" * 50,
            f"Cột số ({len(numeric_columns)}): {', '.join(numeric_columns)}",
            f"Cột chữ ({len(text_columns)}): {', '.join(text_columns)}",
            f"Cột boolean ({len(boolean_columns)}): {', '.join(boolean_columns)}",
            "",
            "Lưu ý về ý nghĩa: OperatingSystems, Browser, Region và TrafficType "
            "được lưu bằng số nhưng là mã phân loại, không phải đại lượng liên tục.",
            "",
            "3. GIÁ TRỊ THIẾU VÀ DÒNG TRÙNG",
            "-" * 50,
            f"Tổng số ô trống: {int(missing_by_column.sum())}",
            f"Số dòng trùng hoàn toàn: {duplicate_count}",
            f"Số dòng còn lại nếu loại trùng: {len(df) - duplicate_count:,}",
            "",
            "4. PHÂN BỐ CỘT KẾT QUẢ REVENUE (DỮ LIỆU GỐC)",
            "-" * 50,
        ]
    )
    for row in target_table.itertuples(index=False):
        lines.append(
            f"{row.label}: {row.count:,} dòng ({row.percentage:.2f}%)"
        )
    lines.extend(
        [
            "",
            "5. CÁC VẤN ĐỀ CẦN XỬ LÝ/LƯU Ý",
            "-" * 50,
            f"- Có {duplicate_count} dòng trùng hoàn toàn; nên loại trước khi chia dữ liệu.",
            "- Không có ô trống." if missing_by_column.sum() == 0 else "- Có ô trống; cần xử lý trước khi huấn luyện.",
            f"- Hai lớp bị mất cân bằng (lớp lớn gấp khoảng {imbalance_ratio:.2f} lần lớp nhỏ); "
            "không nên chỉ dựa vào accuracy.",
            f"- Cột chữ cần mã hóa: {', '.join(text_columns)}.",
            f"- Các cột mã phân loại dạng số cần mã hóa phù hợp: {', '.join(categorical_codes)}.",
            "- Weekend và Revenue là boolean; cần chuyển thành 0/1 khi mô hình yêu cầu.",
            "- Khi chia train/test nên dùng stratify=Revenue để giữ tỷ lệ hai lớp.",
            "",
            f"Bảng phân bố lớp đã lưu tại: {TARGET_TABLE_FILE.name}",
        ]
    )

    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
