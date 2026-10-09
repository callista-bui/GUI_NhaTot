import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
import joblib
from sklearn.ensemble import IsolationForest
from anomaly_detection import detect_anomalies

# Thư mục chứa file price_nhatot.py
BASE_DIR = Path(__file__).resolve().parent

# Open and read file nhatot_price_model.pkl
with open(BASE_DIR / 'nhatot_price_model.pkl', 'rb') as f:
    nhatot_price= joblib.load(f)


st.set_page_config(page_title="Nhà Tốt - Phân tích & Dự đoán giá nhà",page_icon="🏠",layout="wide",)
st.markdown(
    """
    <style>
    /* Màu cam chủ đạo Nhà Tốt */
    :root {
        --primary-color: #F7941D;
    }
    
    /* Tùy chỉnh Tiêu đề chính */
    .main-title {
        color: var(--primary-color);
        font-weight: 700;
        margin-top: -10px;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #555555;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }

    /* Card hiển thị tính năng */
    .feature-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-top: 4px solid var(--primary-color);
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.04);
        height: 100%;
        transition: transform 0.2s ease-in-out;
    }
    .feature-card:hover {
        transform: translateY(-2px);
    }
    .feature-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 8px;
    }
    .feature-desc {
        color: #6b7280;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    </style>
""",
    unsafe_allow_html=True,
)

house = ["Trang chủ","Dự đoán giá nhà", "Phát hiện giá bất thường","Khám phá dữ liệu nhà","Giới thiệu đồ án","Sử dụng các điều khiển"]
choice = st.sidebar.selectbox( "📋 Menu", house, key="main_menu")
st.sidebar.markdown("---")
st.sidebar.caption("💡 *Đồ án Data Science & Machine Learning*")
# Tiêu đề chính của ứng dụng
st.markdown("<h2 class='main-title'>🏠 Nhà Tốt - Phân tích và Dự đoán giá nhà</h2>", unsafe_allow_html=True,)
st.markdown(
    "<div class='sub-title'>"
    "Ứng dụng phân tích dữ liệu, dự đoán giá và phát hiện giá bất thường "
    "bất động sản."
    "</div>",
    unsafe_allow_html=True,
)

# Hình ảnh ở đầu trang
image_path = BASE_DIR / "nhatot.jpg"

if image_path.exists():st.image(image_path, use_container_width=True)
else:
    st.warning("⚠️ Không tìm thấy hình ảnh nhatot.jpg")

# Xử lý các trang theo lựa chọn của Menu
# chọn trang chủ nhớ sửa lại
if choice == "Trang chủ":
    st.subheader("[Trang chủ](https://csc.edu.vn)")
    st.write("""
    ### Chào mừng đến với Ứng dụng Đồ án Data Science

    Ứng dụng tích hợp các mô hình Machine Learning thực tế:
    - 🏠 **Topic 1:** Dự đoán giá nhà 
    - 🏢 **Topic 2:** Phát hiện giá nhà bất thường (Dữ liệu Nhà Tốt)
    """)

elif choice == "Giới thiệu đồ án":
    st.subheader(
        "[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)"
    )
    st.write("""
    ### Có 2 vấn đề chính:
    - **Topic 1:** Dự đoán giá nhà 
    - **Topic 2:** Phát hiện giá nhà bất thường (Dữ liệu Nhà Tốt)
    """)

    # Hiển thị các hình ảnh liên quan đến đồ án
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        img_nhatot = BASE_DIR / "nhatot.jpg"
        if img_nhatot.exists():
            st.image(str(img_nhatot), width=350, caption="Nhà Tốt.vn")
        else:
            st.image(
                "https://static.chotot.com/storage/default/nhatot_logo.png",
                width=300,
                caption="Nhà Tốt.vn",
            )

    with col_img2:
        img_rec = BASE_DIR / "recommend.png"
        if img_rec.exists():
            st.image(str(img_rec), width=350, caption="Recommend Or Not")
        else:
            st.info("📌 *(Hình ảnh `recommend.png`)*")
            
elif choice == "Sử dụng các điều khiển":
    # Sử dụng các điều khiển nhập
    # 1. Text
    st.subheader("1. Text")
    name = st.text_input("Enter your name")
    st.write("Your name is", name)

    # 2. Slider
    st.subheader("2. Slider")
    age = st.slider("How old are you?", 1, 100, 20)
    st.write("I'm", age, "years old.")

    # 3. Checkbox
    st.subheader("3. Checkbox")
    if st.checkbox("I agree"):
        st.write("Great!")

    # 4. Radio
    st.subheader("4. Radio")
    status = st.radio("What is your status?", ("Active", "Inactive"))
    st.write("You are", status)

    # 5. Selectbox
    st.subheader("5. Selectbox")
    occupation = st.selectbox(
        "What is your occupation?", ["Student", "Teacher", "Others"]
    )
    st.write("You are a", occupation)

    # 6. Multiselect
    st.subheader("6. Multiselect")
    location = st.multiselect(
        "Where do you live?", ("Hanoi", "HCM", "Danang", "Hue")
    )
    st.write("You live in", location)

    # 7. File Uploader
    st.subheader("7. File Uploader")
    file = st.file_uploader("Upload your file", type=["csv", "txt"])
    if file is not None:
        st.write(file)

    # 9. Date Input
    st.subheader("9. Date Input")
    date = st.date_input("Pick a date")
    st.write("You picked", date)

    # 10. Time Input
    st.subheader("10. Time Input")
    time = st.time_input("Pick a time")
    st.write("You picked", time)

    # 11. Display JSON
    st.subheader("11. Display JSON")
    json_val = st.text_input("Enter JSON", '{"name": "Alice", "age": 25}')
    st.write("You entered", json_val)

    # 12. Display Raw Code
    st.subheader("12. Display Raw Code")
    code_val = st.text_area("Enter code", "print('Hello, world!')")
    st.write("You entered", code_val)

    # Sử dụng điều khiển submit
    st.subheader("Submit")
    submitted = st.button("Submit")
    if submitted:
        st.write("You submitted the form.")
        # In các thông tin phía trên khi người dùng nhấn nút Submit
        st.write("Your name is", name)
        st.write("I'm", age, "years old.")
        st.write("You are", status)
        st.write("You are a", occupation)
        st.write("You live in", location)
        st.write("You picked", date)
        st.write("You picked", time)
        st.write("You entered", json_val)
        st.write("You entered", code_val)  
elif choice == "Dự đoán giá nhà":
    st.write(
        "##### Dự đoán giá nhà"
    )
    st.write(
        "Giao diện gồm 3 phần: nhập thông tin căn nhà để dự đoán giá, "
        "upload CSV không có giá để dự đoán giá, và upload CSV có giá để "
        "phát hiện bất thường."
    )

    # =========================================================
    # PHẦN 1 - NHẬP THÔNG TIN CĂN NHÀ VÀ DỰ ĐOÁN GIÁ
    # =========================================================
    st.subheader("Phần 1. Nhập thông tin căn nhà → Dự đoán giá")
    st.write("Nhập các thông tin của căn nhà cần dự đoán giá.")

    c1, c2 = st.columns(2)

    with c1:
        house_type = st.selectbox(
            "Loại hình bất động sản",
            ["Nhà ngõ, hẻm", "Nhà mặt phố, mặt tiền", "Nhà phố liền kề","Nhà biệt thự"],
            key="p1_house_type",
        )
        area = st.number_input("Diện tích (m²)", min_value=10.0, value=50.0, step = 1.0)
        
        num_bedrooms = st.selectbox("Số phòng ngủ", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "nhiều hơn 10 phòng"], key="p1_num_bedrooms")

        num_bathrooms = st.selectbox("Số phòng vệ sinh", ["1", "2", "3", "4", "5", "6", "7", "nhiều hơn 7 phòng"], key="p1_num_bathrooms")
        
        # đặc điểm
        special_feature = st.selectbox("Đặc điểm bất động sản(chọn một hoặc nhiều)",
                                       ["Hiện trạng khác", "Hẻm xe hơi","Nhà chưa hoàn công", "Nhà dính quy hoạch / lộ giới", 
                                        "Nhà nát","Nhà nở hậu", "Nhà tóp hậu","Đất chưa chuyển thổ"],
                                       key="p1_special_feature")
        
    with c2:
        district = st.selectbox( "Quận/Huyện",["Gò Vấp", "Bình Thạnh", "Phú Nhuận"],key="p1_district",)
        legal_status = st.selectbox("Tình trạng pháp lý",
            [
                "Đã có sổ",
                "Sổ chung / công chứng vi bằng",
                "Đang chờ sổ",
                "Giấy tờ viết tay",
                "Không có sổ"
            ],
            key="p1_legal_status"
        )
        # số tầng
        num_floors = st.number_input("Số tầng", min_value=1, max_value=44, value=2, step=1, key="p1_num_floors")
        # chiều ngang
        width = st.number_input("Chiều ngang (m)", min_value=2.0, max_value=32.0, value=4.0, step=0.1, key ="p1_width")
        # chiều dài
        length = st.number_input("Chiều dài (m)", min_value=1.4, max_value=24000.0, value=8.0, step=0.1, key ="p1_length")
        # tình trạng nội thất
        furniture_status = st.selectbox("Tình trạng nội thất",["Bàn giao thô","Hoàn thiện cơ bản","Nội thất cao cấp","Nội thất đầy đủ"],
                                        key="p1_furniture_status")       
    if st.button("Dự đoán giá", type="primary", key="p1_predict_manual"):
        st.write("### Thông tin căn nhà đã nhập")

        # An toàn xử lý dac_diem (dù chọn multiselect hay selectbox)
        if isinstance(special_feature, list):
            dac_diem_str = ", ".join(special_feature)
        else:
            dac_diem_str = str(special_feature)
            
        input_data = {
            "loai_hinh": house_type,
            "dien_tich": area,
            "so_phong_ngu": num_bedrooms,
            "so_phong_ve_sinh": num_bathrooms,
            "tong_so_tang": num_floors,
            "quan": district,
            "giay_to_phap_ly": legal_status,
            "tinh_trang_noi_that": furniture_status,
            "dac_diem": dac_diem_str,
            "chieu_ngang": width,
            "chieu_dai": length,
            # "mo_ta": description
        }
        # Chuyển đổi thành DataFrame để đưa vào mô hình dự đoán
        input_df = pd.DataFrame([input_data])
        st.dataframe(input_df, use_container_width=True)
        st.info("Tiếp theo sẽ sử dụng mô hình đã huấn luyện để dự đoán giá.")    
        
        # DỰ ĐOÁN GIÁ 
        # Chạy mô hình để dự đoán giá
        try:
            st.info("🔄 Đang xử lý dữ liệu và dự đoán giá...")

            # Chạy mô hình đã huấn luyện
            prediction = nhatot_price.predict(input_df)

            # Lấy giá trị dự đoán đầu tiên từ mảng
            predicted_price = prediction[0]

            # 4. Hiển thị kết quả định dạng tiền tệ đẹp mắt
            st.success(f"💰 Giá nhà dự đoán: **{predicted_price:,.0f} VND**")

        except Exception as e:
            st.error(
                f"❌ Đã xảy ra lỗi khi dự đoán: {e}. Vui lòng kiểm tra lại Pipeline/Encoder tiền xử lý."
            )
    # =========================================================
    # PHẦN 2 - UPLOAD CSV KHÔNG CÓ GIÁ VÀ DỰ ĐOÁN GIÁ
    # =========================================================
    st.divider()
    st.subheader("Phần 2. Upload CSV không có giá → Dự đoán giá")

    st.write(
        "Upload file CSV chứa thông tin các căn nhà cần dự đoán. "
        "File không cần có các cột `gia_ban`, `gia_m2` hoặc `don_gia` hoặc `gia_ban_vnd`."
    )

    csv_predict = st.file_uploader("Chọn file CSV cần dự đoán giá", type=["csv"], key="p1_csv_predict")

    if csv_predict is not None:
            df_predict = pd.read_csv(csv_predict)

            st.write("Dữ liệu được upload:")
            st.dataframe(df_predict.head(10), use_container_width=True)

            st.write(
                f"Số căn nhà: **{len(df_predict)}** | "
                f"Số thuộc tính: **{len(df_predict.columns)}**"
            )

            cols = [
                "loai_hinh",
                "dien_tich",
                "so_phong_ngu",
                "so_phong_ve_sinh",
                "tong_so_tang",
                "quan",
                "giay_to_phap_ly",
                "tinh_trang_noi_that",
                "dac_diem",
                "chieu_ngang",
                "chieu_dai",
            ]
            price_columns = [
                col for col in cols 
                if col not in df_predict.columns
            ]
            
            # Nếu thiết thiếu các cột bắt buộc
            if price_columns:
                st.error(
                    "❌ File thiếu các cột bắt buộc: "
                    + ", ".join(price_columns)
                )
                st.stop()
            # Kiểm tra giá trị bị thiếu
            empty_columns = {
                col: (
                    df_predict[col].isna()
                    | df_predict[col].astype(str).str.strip().eq("")
                ).sum()
                for col in cols
            }

            # Chỉ lấy các cột có giá trị thiếu
            empty_columns = {
                col: count
                for col, count in empty_columns.items()
                if count > 0
            }

            if empty_columns:
                st.warning("⚠️ File có dữ liệu bị thiếu ở các cột bắt buộc:")

                missing_df = pd.DataFrame(
                    list(empty_columns.items()),
                    columns=["Cột", "Số dòng thiếu"]
                )

                st.dataframe(
                    missing_df,
                    hide_index=True,
                    use_container_width=True
                )

                # st.stop()
            
            # Nút thực hiện dự đoán      
            if st.button(
                "Dự đoán giá cho file CSV",
                type="primary",
                key="p1_predict_csv"
            ):
                # Kiểm tra dữ liệu đầu vào
                try:
                    with st.spinner("Đang thực hiện dự đoán..."):

                        # Chỉ lấy các cột cần thiết đưa vào model
                        X_predict = df_predict[cols]

                        # Chạy mô hình dự đoán
                        prediction = nhatot_price.predict(X_predict)
                
                    # 2. Tạo bản sao kết quả và gán cột giá dự đoán
                    result = df_predict.copy()
                    
                    result["gia_du_doan"] = prediction

                    # Hiển thị kết quả dự đoán cho người dùng
                    st.write("### Kết quả dự đoán")
                    st.dataframe(result, use_container_width=True)

                    # Thống kê giá dự đoán
                    st.write("### Thống kê giá dự đoán")
                    col1, col2, col3 = st.columns(3)

                    # Hàm phụ định dạng tiền tệ gọn hơn (VD: 3.5 Tỷ hoặc 800 Triệu)
                    def format_vnd(amount):
                        if amount >= 1e9:
                            return f"{amount / 1e9:,.2f} Tỷ VNĐ"
                        elif amount >= 1e6:
                            return f"{amount / 1e6:,.0f} Triệu VNĐ"
                        return f"{amount:,.0f} VNĐ"

                    with col1:
                        st.metric("Giá thấp nhất", format_vnd(prediction.min()))

                    with col2:
                        st.metric("Giá trung bình", format_vnd(prediction.mean()))

                    with col3:
                        st.metric("Giá cao nhất", format_vnd(prediction.max()))
                        
                    # Download kết quả dự đoán
                    st.download_button(
                        "Tải kết quả dự đoán(File CSV)",
                        data=result.to_csv(index=False).encode("utf-8-sig"),
                        file_name="ket_qua_du_doan_gia.csv",
                        mime="text/csv",
                        key="p1_download_prediction"
                    )
                except Exception as e:
                    st.error(f"Không thể đọc file CSV: {e}") 
                                 
# elif choice == "Phát hiện giá bất thường":
#     st.write("##### 🚨 Phát hiện giá bất thường")
#     st.write(
#         "Upload file CSV chứa thông tin căn nhà **kèm giá thực tế**. "
#         "Hệ thống sẽ phân tích và xác định các tin đăng có giá bất thường."
#     )
#     # =========================================================
#     # PHẦN 1 - UPLOAD CSV CÓ GIÁ VÀ PHÁT HIỆN BẤT THƯỜNG
#     # =========================================================
#     st.divider()
#     st.subheader("Phần 1. Upload CSV có giá → Phát hiện giá bất thường")

#     csv_anomaly = st.file_uploader(
#         "Chọn file CSV có giá", type=["csv"],key="p1_csv_anomaly" )
   
#     if csv_anomaly is not None:
#         try:
#             df_anomaly = pd.read_csv(csv_anomaly)

#             st.write("Dữ liệu được upload:")
#             st.dataframe(df_anomaly.head(10), use_container_width=True)
            
#             # Tự động tìm cột giá
#             possible_price_cols = [
#                 col
#                 for col in ["gia_ban", "gia_m2", "don_gia", "gia_ban_vnd"]
#                 if col in df_anomaly.columns
#             ]
#             # Nếu không thấy cột ưu tiên, tự tìm các cột dạng số (loại bỏ cột bieu_do_gia nếu có)
#             if not possible_price_cols:
#                 numeric_cols = df_anomaly.select_dtypes(include=["float64", "int64"]).columns.tolist()
#                 possible_price_cols = [col for col in numeric_cols if col != "bieu_do_gia"]
                
#             price_col = st.selectbox(
#                         "Chọn cột giá dùng để phát hiện bất thường",
#                         options=possible_price_cols,
#                         index=0 if possible_price_cols else 0,
#                         key="p1_anomaly_price_col",
#                     )

#             # Chọn phương pháp phát hiện bất thường
#             method = st.selectbox(
#                 "Phương pháp phát hiện bất thường",
#                 [
#                     "Residual-z",
#                     "Vi phạm Min/Max",
#                     "Khoảng P10 - P90",
#                     "Isolation Forest",
#                     "Composite Score"
#                 ],
#                 key="p1_anomaly_method"
#             )
#             #CẤU HÌNH CHO TỪNG PHƯƠNG PHÁP
#             # Tham số cho phương pháp Residual-z
#             if method == "Residual-z":
#                 threshold = st.slider(
#                     "Ngưỡng |Z-score|",
#                     min_value=1.0,
#                     max_value=5.0,
#                     value=3.0,
#                     step=1.0,
#                     key="p1_anomaly_z"
#                 )
         
#             # Tham số cho phương pháp Vi phạm Min/Max
#             elif method == "Vi phạm Min/Max":
#                 st.write(
#                     "Ngưỡng bất thường được tự động xác định "
#                     "theo từng nhóm **Quận/Huyện + Loại hình**."
#                 )   
#                 iqr_multiplier = st.slider(
#                     "Hệ số IQR",
#                     min_value=0.5,
#                     max_value=3.0,
#                     value=1.5,
#                     step=0.1,
#                     key="p1_anomaly_iqr"
#                 )

#             # Tham số cho phương pháp Khoảng P10 - P90
#             elif method == "Khoảng P10 - P90":
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     p_low = st.slider(
#                         "Percentile thấp", 1, 25, 10,key="p1_anomaly_plow"
#                     )
#                 with c2:
#                     p_high = st.slider(
#                         "Percentile cao",75, 99, 90, key="p1_anomaly_phigh"
#                     )

#             # Tham số cho phương pháp Isolation Forest
#             elif method == "Isolation Forest":
#                 contamination = st.slider(
#                     "Tỷ lệ bất thường dự kiến (%)",
#                     1, 20, 5,
#                     key="p1_anomaly_contamination"
#                 )
#             # Tham số cho phương pháp Composite Score
#             else:
#                 st.write("Trọng số Composite Score")
#                 c1, c2 = st.columns(2)
#                 with c1:
#                     w1 = st.slider(
#                         "Residual-z", 0.0, 1.0, 0.25, 0.05, key="p1_w1"
#                     )
#                     w2 = st.slider(
#                         "Min/Max", 0.0, 1.0, 0.25, 0.05, key="p1_w2"
#                     )
#                 with c2:
#                     w3 = st.slider(
#                         "P10-P90", 0.0, 1.0, 0.25, 0.05, key="p1_w3"
#                     )
#                     w4 = st.slider(
#                         "Isolation Forest", 0.0, 1.0, 0.25, 0.05,key="p1_w4"
#                     )
#             # THỰC HIỆN TÍNH TOÁN KHI BẤM NÚT
#             if st.button("Dự đoán bất thường", type="primary", key="p1_detect_anomaly"):
                
#                 if price_col is None:
#                     st.error(
#                         "File CSV phải có ít nhất một cột giá: "
#                         "`gia_ban`, `gia_m2` hoặc `don_gia` hoặc `gia_ban_vnd`."
#                     )
#                 else:
#                     st.info(
#                         f"Đã chọn `{price_col}` và phương pháp "
#                         f"**{method}**. Đây là vị trí để chạy pipeline "
#                         "phát hiện bất thường của project."
#                     )

#                     result = df_anomaly.copy()
                    
#                     price_series = pd.to_numeric(result[price_col], errors="coerce").fillna(0)

#                     # 1. Phương pháp Residual-z
#                     def calc_residual_z(df, y_true):
#                         try:
#                             y_pred = nhatot_price.predict(df)
#                             residuals = y_true - y_pred
#                             mean_res = np.mean(residuals)
#                             std_res = np.std(residuals) + 1e-6
#                             z_scores = np.abs((residuals - mean_res) / std_res)
#                             return z_scores, z_scores > threshold
#                         except Exception as e:
#                             st.warning(
#                                 f"Không thể dự đoán bằng model nhatot_price ({e}), chuyển sang Z-score cơ bản."
#                             )
#                             z_scores = np.abs(
#                                 (y_true - y_true.mean())
#                                 / (y_true.std() + 1e-6)
#                             )
#                             return z_scores, z_scores > threshold

#                     # 2. Phương pháp Min/Max (Tukey IQR theo Nhóm)
           
#                     def calc_tukey_fence(anomaly_pd, price_col, iqr_multiplier=1.5):

#                         GROUP_COLS = ["quan", "loai_hinh"]

#                         anomaly_pd = anomaly_pd.copy()

#                         # Chuyển giá/m² sang numeric trước khi tính Q1, Q3
#                         anomaly_pd[price_col] = pd.to_numeric(
#                             anomaly_pd[price_col],
#                             errors="coerce"
#                         )

#                         # Loại các dòng không có giá hợp lệ
#                         anomaly_pd = anomaly_pd.dropna(
#                             subset=[price_col]
#                         )

#                         # Tính Q1 và Q3 theo Quận + Loại hình
#                         market_bounds = (
#                             anomaly_pd
#                             .groupby(GROUP_COLS)[price_col]
#                             .quantile([0.25, 0.75])
#                             .unstack()
#                             .reset_index()
#                             .rename(
#                                 columns={
#                                     0.25: "q1",
#                                     0.75: "q3"
#                                 }
#                             )
#                         )

#                         # IQR
#                         market_bounds["iqr"] = (
#                             market_bounds["q3"]
#                             - market_bounds["q1"]
#                         )

#                         # Tukey lower fence
#                         market_bounds["gia_m2_floor"] = np.maximum(
#                             0,
#                             market_bounds["q1"]
#                             - iqr_multiplier * market_bounds["iqr"]
#                         )

#                         # Tukey upper fence
#                         market_bounds["gia_m2_ceiling"] = (
#                             market_bounds["q3"]
#                             + iqr_multiplier * market_bounds["iqr"]
#                         )

#                         # Ghép ngưỡng về dữ liệu gốc
#                         result = anomaly_pd.merge(
#                             market_bounds,
#                             on=GROUP_COLS,
#                             how="left"
#                         )

#                         # Đánh dấu bất thường
#                         result["flag_minmax"] = (
#                             (result[price_col] < result["gia_m2_floor"])
#                             |
#                             (result[price_col] > result["gia_m2_ceiling"])
#                         ).astype(int)

#                         result["score_minmax"] = (
#                             result["flag_minmax"].astype(float)
#                         )

#                         return result








#                     # THỰC THI THEO PHƯƠNG PHÁP ĐÃ CHỌN
#                     result = df_anomaly.copy()

#                     if method == "Residual-z":
#                         _, is_anom = calc_residual_z(df_anomaly, price_series)
#                         result["bat_thuong"] = is_anom

#                     elif method == "Vi phạm Min/Max":
#                         result = calc_tukey_fence(
#                             df_anomaly,
#                             price_col="gia_m2",
#                             iqr_multiplier=iqr_multiplier
#                         )

#                         result["bat_thuong"] = (
#                             result["flag_minmax"].astype(int)
#                         )

#                         result["nguong_duoi"] = result["gia_m2_floor"]
#                         result["nguong_tren"] = result["gia_m2_ceiling"]

#                         df_outliers = result[
#                             result["bat_thuong"] == 1
#                         ]

#                         st.success(
#                             f"Phát hiện **{len(df_outliers):,}** bất thường về giá."
#                         )

#                         st.dataframe(
#                             df_outliers,
#                             use_container_width=True
#                         )
#                     elif method == "Khoảng P10 - P90":
#                         _, is_anom, low_v, high_v = calc_p10_p90(
#                             price_series, p_low, p_high
#                         )
#                         result["bat_thuong"] = is_anom
#                         st.info(
#                             f"💡 Khoảng giá chấp nhận được (P{p_low} - P{p_high}): **{low_v:,.2f} → {high_v:,.2f}**"
#                         )
#                     elif method == "Isolation Forest":
#                         _, is_anom = calc_iso_forest(
#                             df_anomaly, price_series, contamination
#                         )
#                         result["bat_thuong"] = is_anom

#                     else:  # Composite Score
#                         s1, _ = calc_residual_z(df_anomaly, price_series)
#                         s2, _, _ = calc_tukey_fence(df_anomaly, price_col=price_col)
#                         s3, _, _, _ = calc_p10_p90(price_series, 10, 90)
#                         s4, _ = calc_iso_forest(df_anomaly, price_series, 5)

#                         s1_norm = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-6)

#                         composite_score = (
#                             w1 * s1_norm + w2 * s2.astype(int) + w3 * s3 + w4 * s4
#                         ) / (w1 + w2 + w3 + w4 + 1e-6)

#                         result["composite_score"] = composite_score
#                         result["bat_thuong"] = composite_score > 0.5
#                     # 3. Phương pháp Quantile P10-P90
#                     def calc_p10_p90(y_true, low_p, high_p):
#                         low_val = y_true.quantile(low_p / 100.0)
#                         high_val = y_true.quantile(high_p / 100.0)
#                         is_anomaly = (y_true < low_val) | (y_true > high_val)
#                         return is_anomaly.astype(int), is_anomaly, low_val, high_val

#                     # 4. Phương pháp Isolation Forest
#                     def calc_iso_forest(df, y_true, cont_rate):
#                         iso = IsolationForest(
#                             contamination=cont_rate / 100.0, random_state=42
#                         )
#                         # Chỉ fit trên các cột dữ liệu số
#                         numeric_df = df.select_dtypes(include=[np.number]).fillna(0)
#                         preds = iso.fit_predict(numeric_df)
#                         is_anomaly = preds == -1
#                         return is_anomaly.astype(int), is_anomaly




#                     # -----------------------------------------------------
#                     # TRỰC QUAN HÓA KẾT QUẢ
#                     # -----------------------------------------------------
#                     num_anomalies = int(result["bat_thuong"].sum())
#                     total_records = len(result)
#                     anomaly_rate = ((num_anomalies / total_records) * 100
#                         if total_records > 0
#                         else 0)

#                     st.write("### 📊 Thống kê kết quả phân tích")
#                     c1, c2, c3 = st.columns(3)
#                     c1.metric("Tổng số tin đăng", f"{total_records:,}")
#                     c2.metric("Số tin giá bất thường", f"{num_anomalies:,}")
#                     c3.metric("Tỷ lệ bất thường", f"{anomaly_rate:.1f}%")

#                     st.write("### 🚨 Danh sách các tin đăng giá bất thường")
#                     anomalies_df = result[result["bat_thuong"]]

#                     if not anomalies_df.empty:
#                         st.dataframe(anomalies_df, use_container_width=True)
#                     else:
#                         st.success( "🎉 Không phát hiện bất kỳ tin đăng bất thường nào với cấu hình hiện tại!")

#                     # Nút tải xuống
#                     st.download_button(
#                         "📥 Tải danh sách tất cả (kèm nhãn bất thường)",
#                         data=result.to_csv(index=False).encode("utf-8-sig"),
#                         file_name="ket_qua_phat_hien_bat_thuong.csv",
#                         mime="text/csv",
#                         key="p1_download_anomaly",
#                     )

#         except Exception as e:
#             st.error(f"❌ Không thể xử lý file CSV: {e}")
    
elif choice == "Phát hiện giá bất thường":

    st.write("##### 🚨 Phát hiện giá bất thường")

    st.write(
        "Upload file CSV chứa thông tin căn nhà "
        "**kèm giá thực tế**. Hệ thống sẽ dự đoán giá "
        "và kết hợp 4 tín hiệu để phát hiện bất thường."
    )

    st.divider()
    st.subheader("Phần 1. Upload CSV có giá → Phát hiện giá bất thường")
    
    # Tạo chức năng up file
    csv_anomaly = st.file_uploader("Chọn file CSV có giá", type=["csv"], key="p1_csv_anomaly")

    if csv_anomaly is not None:

        try:

            df_anomaly = pd.read_csv(csv_anomaly)

            st.write("### 📄 Dữ liệu được upload")
            st.dataframe(df_anomaly.head(10),use_container_width=True)

            # Kiểm tra cột
            required_cols = [
                "gia_ban_vnd",
                "quan",
                "loai_hinh",
                "dien_tich",
                "tong_so_tang",
                "chieu_ngang",
                "chieu_dai",
                "so_phong_ngu",
                "so_phong_ve_sinh"
            ]
            # Tìm các cột thiếu trong file CSV
            missing_cols = [col for col in required_cols if col not in df_anomaly.columns]

            if missing_cols:
                st.error("❌ File CSV thiếu các cột bắt buộc: " + ", ".join(missing_cols))

            else:

                st.success( "✅ File CSV có đầy đủ các cột cần thiết.")

                # Tham số
                st.subheader("⚙️ Cấu hình phát hiện bất thường")

                residual_z_limit = st.slider(
                    "Ngưỡng Residual-Z",
                    1.0,
                    5.0,
                    3.0,
                    0.1,
                    key="p1_residual_z"
                )

                iqr_fence_multiplier = st.slider(
                    "Hệ số Tukey IQR",
                    0.5,
                    3.0,
                    1.5,
                    0.1,
                    key="p1_iqr"
                )

                top_k_percent = st.slider(
                    "Top K (%)",
                    1,
                    20,
                    5,
                    1,
                    key="p1_top_k"
                )

                # -----------------------------------------
                # BUTTON
                # -----------------------------------------

                if st.button("🚨 Phát hiện bất thường",type="primary",key="p1_detect_anomaly"):

                    with st.spinner( "Đang phân tích dữ liệu..."):

                        result = detect_anomalies(
                            df=df_anomaly,
                            model=nhatot_price,
                            residual_z_limit=residual_z_limit,
                            iqr_fence_multiplier=iqr_fence_multiplier,
                            top_k_percent=top_k_percent
                        )
                    # Thống kê

                    total_records = len(result)

                    num_anomalies = (result["anomaly_final"] .eq("Bất thường") .sum())

                    anomaly_rate = (
                        num_anomalies/ total_records* 100
                        if total_records > 0
                        else 0
                    )

                    st.write(
                        "### 📊 Kết quả phân tích"
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric( "Tổng số tin đăng",f"{total_records:,}" )

                    c2.metric( "Tin bất thường",f"{num_anomalies:,}")

                    c3.metric(   "Top K (%)",  f"{anomaly_rate:.1f}%")

                    # Danh sách bất thường

                    st.write("### 🚨 Danh sách tin đăng bất thường" )

                    anomalies_df = result[result["anomaly_final"] == "Bất thường" ]

                    if not anomalies_df.empty:
                        st.dataframe( anomalies_df, use_container_width=True )

                    else:

                        st.success(
                            "🎉 Không phát hiện tin đăng "
                            "bất thường."
                        )

                    # Tải file

                    st.download_button(
                        "📥 Tải kết quả CSV",
                        data=result.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name=(
                            "ket_qua_phat_hien_bat_thuong.csv"
                        ),
                        mime="text/csv",
                        key="p1_download_anomaly"
                    )
        except Exception as e:

            st.error( f"❌ Không thể xử lý file CSV: {e}")




  
    
    
    # btn_predict_single = st.button("🔮 Dự đoán giá căn nhà này")
    # if btn_predict_single:
    #     st.success(
    #         "Kết quả dự đoán: **3.5 Tỷ VNĐ** (Ví dụ kết quả từ model)"
    #     )

    # st.markdown("---")         
    # st.image(BASE_DIR / "nhatot.jpg",width=500,caption="Dự án Nhà Tốt")







# Thư mục chứa file price_nhatot.py
# BASE_DIR = Path(__file__).resolve().parent

# Đường dẫn đến dữ liệu
# DATA_PATH = BASE_DIR / "data_cleaned.csv"
# # Đọc dữ liệu sản phẩm
# df_houses = pd.read_csv(DATA_PATH)

# # Tạo ID cho từng căn nhà
# df_houses["house_id"] = df_houses.index

# # Tạo 10 căn nhà ngẫu nhiên
# if 'random_houses' not in st.session_state:
#     st.session_state.random_houses = df_houses.sample(n=10, random_state=42).copy()
    
# # Open and read file nhatot_price_model.pkl
# with open(BASE_DIR / 'nhatot_price_model.pkl', 'rb') as f:
#     nhatot_price= joblib.load(f)
    
# ###### Giao diện Streamlit ######

# st.image(
#     BASE_DIR / "nhatot.jpg",
#     use_container_width=True
# )
   
# Tạo dropdown
# house_options = [
#     (row["tieu_de"], row["house_id"])
#     for _, row in st.session_state.random_houses.iterrows()
# ]

# selected_house_option = st.selectbox(
#     "Tìm nhà",
#     options=house_options,
#     format_func=lambda x: x[0]
# )

# Lưu căn nhà được chọn
# st.session_state.selected_id = selected_house_option[1]

# Lấy thông tin căn nhà
# selected_house_data = st.session_state.random_houses[
#     st.session_state.random_houses["house_id"]
#     == st.session_state.selected_id
# ].iloc[0]

# Hiển thị
# st.write("### Thông tin căn nhà")
# st.write("**Tiêu đề:**", selected_house_data["tieu_de"])
# st.write("**Giá bán:**", selected_house_data["gia_ban"])
# st.write("**Diện tích:**", selected_house_data["dien_tich"])
# st.write("**Địa chỉ:**", selected_house_data["dia_chi"])
# st.write("**Tổng quan về căn nhà:**", selected_house_data["Tổng quan về căn nhà"])
