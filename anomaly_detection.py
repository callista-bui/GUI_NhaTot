import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


def detect_anomalies(
    df,
    model,
    residual_z_limit=3.0,
    iqr_fence_multiplier=1.5,
    top_k_percent=5.0
):
    """
    Phát hiện tín hiệu bất thường về giá bất động sản.
    Parameters
    ----------
    df : pd.DataFrame Dữ liệu đầu vào, phải có các cột cần thiết.

    model : sklearn model/pipeline Model dự đoán giá bán VND.

    residual_z_limit : float
        Ngưỡng Residual-Z.

    iqr_fence_multiplier : float
        Hệ số Tukey Fence.

    top_k_percent : float
        Tỷ lệ top % được đánh dấu bất thường.

    Returns
    -------
    anomaly_pd : pd.DataFrame
        DataFrame chứa toàn bộ kết quả.
    """

    # =========================================================
    # 1. DỰ ĐOÁN GIÁ BÁN
    # =========================================================

    df_anomaly = df.copy()

    pred_anomaly = model.predict(df_anomaly)

    df_anomaly["gia_du_doan_vnd"] = pred_anomaly

    df_anomaly["gia_thuc_te_vnd"] = ( df_anomaly["gia_ban_vnd"])

    # Chuyển sang tỷ đồng
    df_anomaly["gia_thuc_te"] = (df_anomaly["gia_thuc_te_vnd"] / 1e9)
    df_anomaly["prediction"] = (df_anomaly["gia_du_doan_vnd"] / 1e9)

    # =========================================================
    # 2. CHUẨN BỊ DỮ LIỆU
    # =========================================================

    cols_for_anomaly = [
        "quan",
        "loai_hinh",
        "dien_tich",
        "gia_thuc_te",
        "prediction",
        "tong_so_tang",
        "chieu_ngang",
        "chieu_dai",
        "so_phong_ngu",
        "so_phong_ve_sinh"
    ]

    missing_cols = [
        col
        for col in cols_for_anomaly
        if col not in df_anomaly.columns
    ]

    if missing_cols:
        raise ValueError(
            "Thiếu các cột cần thiết: "
            + ", ".join(missing_cols)
        )

    anomaly_pd = (
        df_anomaly[cols_for_anomaly]
        .dropna(
            subset=[
                "gia_thuc_te",
                "prediction",
                "dien_tich"
            ]
        )
        .copy()
    )

    # Điền giá trị phân loại
    anomaly_pd["quan"] = ( anomaly_pd["quan"].fillna("Không rõ"))

    anomaly_pd["loai_hinh"] = ( anomaly_pd["loai_hinh"].fillna("Không rõ"))

    # =========================================================
    # 3. GIÁ / M² THỰC TẾ
    # =========================================================

    anomaly_pd["gia_m2_thuc_te"] = np.where(
        anomaly_pd["dien_tich"] > 0,
        anomaly_pd["gia_thuc_te"]
        / anomaly_pd["dien_tich"],
        np.nan
    )

    # 4. TÍN HIỆU 1: RESIDUAL-Z
    
    anomaly_pd["residual"] = (anomaly_pd["gia_thuc_te"]- anomaly_pd["prediction"])

    anomaly_pd["price_gap_pct"] = np.where( anomaly_pd["prediction"] > 0,anomaly_pd["residual"]/ anomaly_pd["prediction"] * 100,np.nan)

    resid_std = ( anomaly_pd["residual"].std(ddof=0))

    if resid_std == 0 or pd.isna(resid_std):
        anomaly_pd["residual_zscore"] = 0.0
    else:

        anomaly_pd["residual_zscore"] = ( anomaly_pd["residual"]- anomaly_pd["residual"].mean()) / resid_std

    anomaly_pd["flag_residual"] = (anomaly_pd["residual_zscore"] .abs()> residual_z_limit).astype(int)

    anomaly_pd["score_residual"] = np.clip(anomaly_pd["residual_zscore"].abs()/ residual_z_limit, 0, 1)

    # =========================================================
    # 5. TÍN HIỆU 2: TUKEY FENCE
    # =========================================================

    GROUP_COLS = [
        "quan",
        "loai_hinh"
    ]

    group_price = (
        anomaly_pd
        .groupby(GROUP_COLS)["gia_m2_thuc_te"]
    )

    market_bounds = (
        group_price
        .quantile(
            [0.10, 0.25, 0.75, 0.90]
        )
        .unstack()
        .reset_index()
        .rename(
            columns={
                0.10: "gia_m2_p10",
                0.25: "q1",
                0.75: "q3",
                0.90: "gia_m2_p90"
            }
        )
    )

    market_bounds["iqr"] = (
        market_bounds["q3"]
        - market_bounds["q1"]
    )

    market_bounds["gia_m2_floor"] = np.maximum(
        0,
        market_bounds["q1"]
        - iqr_fence_multiplier
        * market_bounds["iqr"]
    )

    market_bounds["gia_m2_ceiling"] = (
        market_bounds["q3"]
        + iqr_fence_multiplier
        * market_bounds["iqr"]
    )

    anomaly_pd = anomaly_pd.merge(
        market_bounds,
        on=GROUP_COLS,
        how="left"
    )

    anomaly_pd["flag_minmax"] = (
        (
            anomaly_pd["gia_m2_thuc_te"]
            < anomaly_pd["gia_m2_floor"]
        )
        |
        (
            anomaly_pd["gia_m2_thuc_te"]
            > anomaly_pd["gia_m2_ceiling"]
        )
    ).astype(int)

    anomaly_pd["score_minmax"] = (
        anomaly_pd["flag_minmax"]
        .astype(float)
    )

    # =========================================================
    # 6. TÍN HIỆU 3: P10 - P90
    # =========================================================

    anomaly_pd["flag_p10_p90"] = (
        (
            anomaly_pd["gia_m2_thuc_te"]
            < anomaly_pd["gia_m2_p10"]
        )
        |
        (
            anomaly_pd["gia_m2_thuc_te"]
            > anomaly_pd["gia_m2_p90"]
        )
    ).astype(int)

    distance_below = (
        anomaly_pd["gia_m2_p10"]
        - anomaly_pd["gia_m2_thuc_te"]
    ).clip(lower=0)

    distance_above = (
        anomaly_pd["gia_m2_thuc_te"]
        - anomaly_pd["gia_m2_p90"]
    ).clip(lower=0)

    anomaly_pd["distance_p10_p90"] = (
        distance_below
        + distance_above
    )

    distance_min = (
        anomaly_pd["distance_p10_p90"].min()
    )

    distance_max = (
        anomaly_pd["distance_p10_p90"].max()
    )

    if (
        pd.isna(distance_min)
        or distance_max == distance_min
    ):

        anomaly_pd["score_p10_p90"] = 0.0

    else:

        anomaly_pd["score_p10_p90"] = (
            anomaly_pd["distance_p10_p90"]
            - distance_min
        ) / (
            distance_max
            - distance_min
        )

    # =========================================================
    # 7. TÍN HIỆU 4: ISOLATION FOREST
    # =========================================================

    iso_features = [
        "gia_m2_thuc_te",
        "dien_tich",
        "tong_so_tang",
        "chieu_ngang",
        "chieu_dai"
    ]

    iso_df = (
        anomaly_pd[iso_features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    iso_df = (
        iso_df
        .fillna(iso_df.median())
        .fillna(0)
    )

    iso_model = IsolationForest(
        random_state=42,
        contamination="auto"
    )

    anomaly_pd["isolation_label"] = (
        iso_model.fit_predict(iso_df)
    )

    anomaly_pd["isolation_raw"] = (
        -iso_model.decision_function(iso_df)
    )

    anomaly_pd["score_isolation"] = (
        MinMaxScaler()
        .fit_transform(
            anomaly_pd[
                ["isolation_raw"]
            ]
        )
        .ravel()
    )

    # =========================================================
    # 8. COMPOSITE SCORE
    # =========================================================

    weights = {
        "score_residual": 0.25,
        "score_minmax": 0.25,
        "score_p10_p90": 0.25,
        "score_isolation": 0.25
    }

    assert np.isclose(
        sum(weights.values()),
        1.0
    )

    anomaly_pd["weighted_anomaly_score"] = (
        sum(
            weights[name]
            * anomaly_pd[name]
            for name in weights
        )
        * 100
    ).round(2)

    # =========================================================
    # 9. TOP 5%
    # =========================================================

    threshold = (
        anomaly_pd[
            "weighted_anomaly_score"
        ].quantile(
            1 - top_k_percent / 100
        )
    )

    anomaly_pd["anomaly_final"] = np.where(
        anomaly_pd[
            "weighted_anomaly_score"
        ] >= threshold,
        "Bất thường",
        "Bình thường"
    )

    # =========================================================
    # 10. PHÂN LOẠI NGUYÊN NHÂN
    # =========================================================

    def classify_anomaly(row):

        if row["anomaly_final"] == "Bình thường":
            return (
                "Bình thường",
                "Điểm tổng hợp dưới ngưỡng top-k"
            )

        reasons = []

        if row["flag_residual"]:
            reasons.append(
                f"Residual-Z vượt ±{residual_z_limit:g}"
            )

        if row["flag_minmax"]:
            reasons.append(
                "Đơn giá m² ngoài khung "
                "Tukey Min/Max của nhóm"
            )

        if row["flag_p10_p90"]:
            reasons.append(
                "Đơn giá m² ngoài P10-P90 của nhóm"
            )

        if row["isolation_label"] == -1:
            reasons.append(
                "Isolation Forest gắn nhãn outlier"
            )

        if (
            row["residual"] > 0
            and row["flag_residual"]
        ):
            anomaly_type = (
                "Giá cao hơn dự đoán"
            )

        elif (
            row["residual"] < 0
            and row["flag_residual"]
        ):
            anomaly_type = (
                "Giá thấp hơn dự đoán"
            )

        elif (
            row["gia_m2_thuc_te"]
            > row["gia_m2_p90"]
        ):
            anomaly_type = "Đơn giá cao"

        elif (
            row["gia_m2_thuc_te"]
            < row["gia_m2_p10"]
        ):
            anomaly_type = "Đơn giá thấp"

        else:
            anomaly_type = (
                "Tổ hợp đặc trưng bất thường"
            )

        if not reasons:
            reasons.append(
                f"Thuộc top {top_k_percent:g}% "
                "điểm tổng hợp; các tín hiệu riêng "
                "chưa vượt cờ nhị phân"
            )

        return (
            anomaly_type,
            "; ".join(reasons)
        )

    classification = anomaly_pd.apply(
        classify_anomaly,
        axis=1,
        result_type="expand"
    )

    anomaly_pd["anomaly_type"] = (
        classification[0]
    )

    anomaly_pd["anomaly_reason"] = (
        classification[1]
    )

    return anomaly_pd