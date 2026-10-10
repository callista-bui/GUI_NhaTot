import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os
import joblib
from sklearn.ensemble import IsolationForest
from anomaly_detection import detect_anomalies
import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns

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
    /* Màu chữ Label "📋 Menu" */
    [data-testid="stSidebar"] div[class*="stSelectbox"] label p {
        color: #000000 !important;
        font-weight: 600;
    }

    /* Màu chữ các lựa chọn trong ô Selectbox */
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #000000 !important;
    }
    /* Màu chữ caption trong Sidebar */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        color: #000000 !important;
    }
    
    </style>
""",
    unsafe_allow_html=True,
)

# 1. Cấu hình trang
st.set_page_config(
    page_title="Nhà Tốt - AI Real Estate App",
    page_icon="🏠",
    layout="wide"
)

house_options = ["Trang chủ","Dự đoán giá nhà", "Phát hiện giá bất thường"]

house = st.sidebar.selectbox( "📋 Menu", house_options, key="main_menu")
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #E6F2FC;
        opacity: 1;
    }

    [data-testid="stSidebar"] * {
        color: #191a1c;
    }
    </style>
    """, unsafe_allow_html=True)

# Tiêu đề chính của ứng dụng
st.markdown("<h2 class='main-title'>🏠 Nhà Tốt - Phân tích và Dự đoán giá nhà</h2>", unsafe_allow_html=True,)
# st.markdown(
#     "<div class='sub-title'>"
#     "Ứng dụng phân tích dữ liệu, dự đoán giá và phát hiện giá bất thường "
#     "bất động sản."
#     "</div>",
#     unsafe_allow_html=True,
# )

# Hình ảnh ở đầu trang
image_path = BASE_DIR / "nhatot.jpg"

if image_path.exists():st.image(image_path, use_container_width=150)
else:
    st.warning("⚠️ Không tìm thấy hình ảnh nhatot.jpg")

import streamlit as st

st.sidebar.markdown("---")

# Tạo khoảng trống trước phần thông tin
st.sidebar.markdown(
    """
    <div style="height: 35vh;"></div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("**Thông tin thành viên**")

st.sidebar.caption(
    "Lê Thị Hà My\n\n"
    "📧 lehamy.yds@gmail.com\n\n"
    "Bùi Thị Thư\n\n"
    "📧 buithithu.ntt@gmail.com"
)

st.sidebar.caption("💡 *Đồ án Data Science & Machine Learning*")
if house == "Trang chủ":
    # --- HEADER BANNER TRANG CHỦ ---
    # st.title("🚀 Đồ án Data Science")
    st.markdown("Đơn vị đào tạo: [Trung tâm Tin học - ĐH KHTN](https://csc.edu.vn)")
    
    st.info("Chào mừng bạn đến với ứng dụng tích hợp các mô hình Machine Learning thực tế!")
    
    st.markdown(
        """
        <h4 style="margin-bottom: 4px;">🎯 Hai chức năng chính</h4>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Hệ thống hỗ trợ người dùng tham khảo giá nhà và nhận diện "
        "những tin đăng có mức giá khác thường."
    )

    col1, col2 = st.columns(2, gap="medium")
    # =========================================================
    # CHỨC NĂNG 1
    # =========================================================
    with col1:
        st.markdown(
            "<h5 style='margin-bottom: 6px;'>📈 Dự đoán giá nhà</h5>",
            unsafe_allow_html=True
        )

        st.write(
            "Ước tính giá bán của một căn nhà dựa trên các thông tin "
            "như vị trí, diện tích, số phòng và đặc điểm của bất động sản."
        )

        st.markdown(
            "<div style='margin: 8px 0; border-top: 1px solid #ddd;'></div>",
            unsafe_allow_html=True
        )

        st.markdown("**Cách thực hiện & Đánh giá**")
        st.caption("Mô hình Random Forest, đánh giá bằng MAE, RMSE và R²")

    # =========================================================
    # CHỨC NĂNG 2
    # =========================================================
    with col2:
        st.markdown(
            "<h5 style='margin-bottom: 6px;'>🛡️ Phát hiện giá bất thường</h5>",
            unsafe_allow_html=True
        )

        st.write(
            "Kiểm tra giá của tin đăng để tìm những căn nhà có giá "
            "cao hoặc thấp bất thường so với mức giá phổ biến của "
            "các căn nhà tương tự."
        )

        st.markdown(
            "<div style='margin: 8px 0; border-top: 1px solid #ddd;'></div>",
            unsafe_allow_html=True
        )

        st.markdown("**Cách thực hiện & Đánh giá**")
        st.caption("So sánh nhiều tiêu chí về giá, đánh giá bằng Anomaly Score")

    # =========================================================
    # QUY TRÌNH
    # =========================================================

    st.markdown(
        "<div style='margin: 12px 0 6px; border-top: 1px solid #ddd;'></div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h4 style="margin-bottom: 2px;">🔄 Quy trình xây dựng hệ thống</h4>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Từ dữ liệu đầu vào đến hệ thống ứng dụng thực tế"
    )

    p1, p2, p3, p4, p5 = st.columns(5, gap="small")

    steps = [
        ("01", "📂", "Dữ liệu"),
        ("02", "🧹", "Làm sạch"),
        ("03", "🤖", "Xây dựng"),
        ("04", "📊", "Đánh giá"),
        ("05", "🚀", "Ứng dụng")
    ]

    for col, (num, icon, title) in zip(
        [p1, p2, p3, p4, p5], steps
    ):
        with col:
            st.markdown(f"**{num}**")
            st.markdown(
                f"<div style='font-size: 24px; margin: 0;'>{icon}</div>",
                unsafe_allow_html=True
            )
            st.caption(title) 
            
elif house == "Dự đoán giá nhà":
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
    st.header("🏠 Dự Đoán Giá Nhà Bất Động Sản", divider="rainbow")
    st.caption("Nhập các thông số chi tiết của căn nhà bên dưới để mô hình Machine Learning tính toán mức giá ước tính.")

    st.write("") # Dòng trống tạo khoảng nghỉ

    c1, c2 = st.columns(2)

    with c1:
        house_type = st.selectbox(
            "Loại hình bất động sản",
            ["Nhà ngõ, hẻm", "Nhà mặt phố, mặt tiền", "Nhà phố liền kề","Nhà biệt thự"],
            key="p1_house_type",
        )
        district = st.selectbox( "Quận/Huyện",["Gò Vấp", "Bình Thạnh", "Phú Nhuận"],key="p1_district",)
        
        area = st.number_input("Diện tích (m²)", min_value=10.0, value=50.0, step = 1.0)
        
        # số tầng
        num_floors = st.number_input("Số tầng", min_value=1, max_value=44, value=2, step=1, key="p1_num_floors")
        
    with c2:
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
        num_bedrooms = st.selectbox("Số phòng ngủ", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "nhiều hơn 10 phòng"], key="p1_num_bedrooms")

        num_bathrooms = st.selectbox("Số phòng vệ sinh", ["1", "2", "3", "4", "5", "6", "7", "nhiều hơn 7 phòng"], key="p1_num_bathrooms")
        
        # đặc điểm
        special_feature = st.selectbox("Đặc điểm bất động sản(chọn một hoặc nhiều)",
                                       ["Hiện trạng khác", "Hẻm xe hơi","Nhà chưa hoàn công", "Nhà dính quy hoạch / lộ giới", 
                                        "Nhà nát","Nhà nở hậu", "Nhà tóp hậu","Đất chưa chuyển thổ"],
                                       key="p1_special_feature")
    
    
         
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
            "dac_diem": dac_diem_str,
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
    # ===== [SỬ DỤNG .get() VÀ TRUY XUẤT AN TOÀN] =====
    # st.markdown("---")
    # st.subheader("📋 Danh Sách Căn Nhà Đã Dự Đoán & Lưu")

    # # Sử dụng .get() phòng trường hợp chưa khởi tạo để không bao giờ bị KeyError
    # saved_list = st.session_state.get("saved_houses", [])

    # if len(saved_list) > 0:
    #     saved_df = pd.DataFrame(saved_list)
    #     st.dataframe(saved_df, use_container_width=True)

    #     col_dl1, col_dl2 = st.columns([1, 1])
    #     with col_dl1:
    #         csv_data = saved_df.to_csv(index=False).encode("utf-8-sig")
    #     st.download_button(
    #         label="📥 Tải danh sách đã lưu (CSV)",
    #         data=csv_data,
    #         file_name="danh_sach_nha_da_du_doan.csv",
    #         mime="text/csv",
    #         key="download_saved_houses",
    #     )
    #     with col_dl2:
    #         if st.button("🗑️ Xóa lịch sử đã lưu", key="clear_saved_houses"):
    #             st.session_state.saved_houses = []
    #         st.rerun()
    # else:
    #     st.info(
    #     "Chưa có căn nhà nào được lưu. Nhấn nút 'Dự đoán giá' để lưu thông tin"
    #     " vào danh sách."
    # )

    # =========================================================
    # PHẦN 2 - UPLOAD CSV KHÔNG CÓ GIÁ VÀ DỰ ĐOÁN GIÁ
    # =========================================================
    st.divider()
    st.subheader("📂 Dự Đoán Giá Hàng Loạt Từ File CSV", divider="green")

    st.write(
        "Upload file CSV chứa thông tin các căn nhà cần dự đoán. "
    )

    csv_predict = st.file_uploader("Chọn file CSV cần dự đoán giá", type=["csv"], key="p1_csv_predict")

    if csv_predict is not None:
            try:
                df_predict = pd.read_csv(csv_predict)
            except Exception as e:
                st.error(f"❌ Không thể đọc file CSV: {e}")
                st.stop()

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
                "dac_diem",
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
                try:
                    with st.spinner("🚀 Đang thực hiện dự đoán..."):

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
                        "📥 Tải kết quả dự đoán(File CSV)",
                        data=result.to_csv(index=False).encode("utf-8-sig"),
                        file_name="ket_qua_du_doan_gia.csv",
                        mime="text/csv",
                        type="secondary",
                        use_container_width=True,
                        key="p1_download_prediction"
                    )
                            
                    # ==================== BẮT ĐẦU PHẦN THÊM BIỂU ĐỒ ====================
                    import plotly.express as px

                    st.write("### 📊 Trực quan hóa kết quả dự đoán")

                    tab_dist, tab_scatter = st.tabs(
                        ["Phân phối giá dự đoán", "Giá theo diện tích"]
                    )

                    with tab_dist:
                        # Chuyển đổi giá sang đơn vị Tỷ VNĐ cho gọn
                        df_chart = result.copy()
                        df_chart["gia_ty"] = df_chart["gia_du_doan"] / 1e9

                        # Vẽ biểu đồ phân phối với Plotly Express (an toàn, không lỗi version)
                        fig_dist = px.histogram(
                            df_chart,
                            x="gia_ty",
                            nbins=30,
                            marginal="box",  # Hiện biểu đồ Boxplot nhỏ ở mép trên để xem cực trị / median
                            title="Phân phối giá dự đoán bất động sản",
                            labels={
                                "gia_ty": "Giá dự đoán (Tỷ VNĐ)",
                                "count": "Số lượng căn nhà",
                            },
                            color_discrete_sequence=["#1f77b4"],
                        )

                        fig_dist.update_layout(
                            xaxis_title="Giá (Tỷ VNĐ)",
                            yaxis_title="Số lượng căn nhà",
                            bargap=0.05,
                        )

                        st.plotly_chart(
                            fig_dist,
                            use_container_width=True,
                            alt="Phân phối giá dự đoán",
                        )

                    with tab_scatter:
                        if "dien_tich" in result.columns:
                            fig_scatter = px.scatter(
                                df_chart,
                                x="dien_tich",
                                y="gia_ty",
                                labels={
                                    "dien_tich": "Diện tích (m²)",
                                    "gia_ty": "Giá dự đoán (Tỷ VNĐ)",
                                },
                                title="Mối quan hệ giữa Diện tích và Giá",
                                hover_data=["quan"]
                                if "quan" in df_chart.columns
                                else None,
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.info("Không tìm thấy cột 'dien_tich' để vẽ biểu đồ.")
                except Exception as e:
                    st.error(f"Không thể đọc file CSV: {e}") 
                    
elif house == "Phát hiện giá bất thường":
  
# =========================================================
    # KIỂM TRA BẤT THƯỜNG BẤT ĐỘNG SẢN (ANOMALY DETECTION)
    # =========================================================
    st.markdown("---")
    st.header("🚨 Kiểm Tra Tin Bán Nhà Bất Thường", divider="red")
    st.caption(
        "Nhập thông tin kèm **giá rao bán thực tế** hoặc upload file CSV có giá để phát hiện các căn nhà có mức giá bất thường (quá rẻ hoặc ngáo giá)."
    )

    tab_manual, tab_csv = st.tabs(["📝 Nhập tay 1 căn nhà", "📁 Upload file CSV"])

    # ---------------------------------------------------------
    # TAB 1: NHẬP TAY 1 CĂN NHÀ ĐỂ KIỂM TRA
    # ---------------------------------------------------------
    with tab_manual:
        with st.form("anomaly_manual_form"):
            ac1, ac2 = st.columns(2)
            with ac1:
                a_house_type = st.selectbox(
                    "Loại hình bất động sản",
                    ["Nhà ngõ, hẻm", "Nhà mặt phố, mặt tiền", "Nhà phố liền kề", "Nhà biệt thự"],
                    key="p3_house_type"
                )
                a_district = st.selectbox("Quận/Huyện", ["Gò Vấp", "Bình Thạnh", "Phú Nhuận"], key="p3_district")
                a_area = st.number_input("Diện tích (m²)", min_value=10.0, value=50.0, step=1.0, key="p3_area")
                a_num_floors = st.number_input("Số tầng", min_value=1, max_value=44, value=2, step=1, key="p3_num_floors")

            with ac2:
                a_legal_status = st.selectbox(
                    "Tình trạng pháp lý",
                    ["Đã có sổ", "Sổ chung / công chứng vi bằng", "Đang chờ sổ", "Giấy tờ viết tay", "Không có sổ"],
                    key="p3_legal_status"
                )
                a_num_bedrooms = st.selectbox("Số phòng ngủ", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "nhiều hơn 10 phòng"], key="p3_num_bedrooms")
                a_num_bathrooms = st.selectbox("Số phòng vệ sinh", ["1", "2", "3", "4", "5", "6", "7", "nhiều hơn 7 phòng"], key="p3_num_bathrooms")
                a_special_feature = st.selectbox(
                    "Đặc điểm bất động sản",
                    ["Hiện trạng khác", "Hẻm xe hơi", "Nhà chưa hoàn công", "Nhà dính quy hoạch / lộ giới", "Nhà nát", "Nhà nở hậu", "Nhà tóp hậu", "Đất chưa chuyển thổ"],
                    key="p3_special_feature"
                )

            st.write("---")
            # Ô nhập giá rao bán thực tế để so sánh / kiểm tra
            actual_price_ty = st.number_input(
                "💰 **Giá rao bán thực tế ( Tỷ VND)**",
                min_value=0.1,
                max_value=1000.0,
                value=5.0,
                step=0.1,
                format="%.1f",
                key="p3_actual_price"
            )

            btn_check_anomaly = st.form_submit_button("🔍 Kiểm tra bất thường", type="primary")

        if btn_check_anomaly:   
            # Chuẩn hóa dữ liệu đầu vào
            actual_price_vnd = actual_price_ty * 1_000_000_000
            a_dac_diem_str = ", ".join(a_special_feature) if isinstance(a_special_feature, list) else str(a_special_feature)
            
            a_input_data = {
                "loai_hinh": a_house_type,
                "dien_tich": a_area,
                "so_phong_ngu": a_num_bedrooms,
                "so_phong_ve_sinh": a_num_bathrooms,
                "tong_so_tang": a_num_floors,
                "quan": a_district,
                "giay_to_phap_ly": a_legal_status,
                "dac_diem": a_dac_diem_str,
                "gia_ban": actual_price_vnd  # Cột giá thực tế
            }

            a_input_df = pd.DataFrame([a_input_data])

            try:
                # 1. Dự đoán giá định giá thị trường (dùng model định giá nếu có, hoặc tính theo đơn giá m2)
                if "nhatot_price" in globals() and hasattr(nhatot_price, "predict"):
                    pred_price = float(nhatot_price.predict(a_input_df.drop(columns=["gia_ban"]))[0])
                else:
                    st.error("Chưa có mô hình định giá thị trường.")
                    st.stop()
                    
                # 2. TÍNH ĐỘ LỆCH PHẦN TRĂM VÀ ĐIỂM BẤT THƯỜNG (Không cần Anomaly Model)
                # Lệch % = (Giá rao bán - Giá định giá) / Giá định giá
                diff_ratio = (actual_price_vnd - pred_price) / pred_price
                diff_pct = abs(diff_ratio) * 100

                # Hiển thị kết quả so sánh
                col_res1, col_res2 = st.columns(2)
                col_res1.metric("Giá rao bán thực tế", f"{actual_price_vnd / 1_000_000_000:.1f} Tỷ VND")
                col_res2.metric(
                        "Giá định giá tham chiếu",
                        f"{pred_price / 1_000_000_000:.1f} Tỷ VND",
                        delta=f"{diff_ratio * 100:+.1f}% so với thị trường",
                        delta_color="inverse",  # Đỏ nếu đắt hơn, Xanh nếu rẻ hơn
                    )

                st.write("---")

                # 3. ĐÁNH GIÁ BẤT THƯỜNG THEO NGƯỠNG QỦY TẮC (THRESHOLD)
                # Ngưỡng: Lệch trên 35% được coi là Bất thường
                if diff_pct > 35:
                    st.error("🚨 **CẢNH BÁO: TIN BÁN NHÀ BẤT THƯỜNG!**")

                    if actual_price_vnd < pred_price:
                        st.warning(
                                f"⚠️ **Rủi ro lừa đảo / Pháp lý:** Giá rao bán rẻ hơn thị trường"
                                f" **{diff_pct:.1f}%**. "
                                "Cần kiểm tra kỹ quy hoạch, tranh chấp, sổ đỏ hoặc nguy cơ lừa đảo"
                                " tiền cọc."
                            )
                    else:
                        st.warning(
                                f"⚠️ **Cảnh báo ngáo giá:** Giá rao bán cao hơn thị trường"
                                f" **{diff_pct:.1f}%**. "
                                "Mức giá này cao vượt trội so với mặt bằng chung cùng khu vực."
                        )
                else:
                    st.success(
                        f"✅ **TIN BÁN NHÀ BÌNH THƯỜNG:** Mức giá hợp lý (độ lệch"
                        f" **{diff_pct:.1f}%** nằm trong khoảng cho phép ±35%)."
                    )

            except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra: {e}")

    # ---------------------------------------------------------
    # TAB 2: UPLOAD FILE CSV CÓ GIÁ ĐỂ KIỂM TRA HÀNG LOẠT
    # ---------------------------------------------------------
    with tab_csv:
        st.subheader("📁 Tải lên file CSV chứa danh sách căn nhà (có cột giá)")
        
        # Tạo chức năng up file
        csv_anomaly = st.file_uploader("Chọn file CSV có giá", type=["csv"], key="p1_csv_anomaly")
    if csv_anomaly is not None:
        try:

            df_anomaly = pd.read_csv(csv_anomaly)

            st.write("📄 Dữ liệu được upload")
            st.write("📋 Xem trước dữ liệu Upload")
            st.dataframe(df_anomaly.head(10),use_container_width=True)

            # Kiểm tra cột
            required_cols = [
                "gia_ban_vnd",
                "quan",
                "loai_hinh",
                "dien_tich",
                "tong_so_tang",
                "so_phong_ngu",
                "so_phong_ve_sinh",
                "giay_to_phap_ly",
            ]
            # Tìm các cột thiếu trong file CSV
            missing_cols = [col for col in required_cols if col not in df_anomaly.columns]

            if missing_cols:
                st.error("❌ File CSV thiếu các cột bắt buộc: " + ", ".join(missing_cols))

            else:

                st.success( "✅ File CSV có đầy đủ các cột cần thiết.")

                if st.button("🚨 Phát hiện bất thường",type="primary",key="p1_detect_anomaly"):

                    with st.spinner( "Đang phân tích dữ liệu..."):

                        result = detect_anomalies(
                            df=df_anomaly,
                            model=nhatot_price,
                        )
                        st.session_state["p1_anomaly_result"] = result
                        
                    # Thống kê
                if "p1_anomaly_result" in st.session_state:
                    result = st.session_state["p1_anomaly_result"]
                    
                    total_records = len(result)

                    num_anomalies = (result["anomaly_final"] .eq("Bất thường") .sum())

                    anomaly_rate = (
                        num_anomalies/ total_records* 100
                        if total_records > 0
                        else 0
                    )

                    st.write("### 📊 Kết quả phân tích")

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
                   
                    # Biểu đồ 1: Giá thực tế vs Giá dự đoán
                    # st.write("### 📈 Biểu đồ Giá Thực Tế vs. Giá Dự Đoán (Model AI)")   

                    # fig1 = px.scatter(
                    #     result,
                    #     x="prediction",
                    #     y="gia_thuc_te",
                    #     color="anomaly_final",  # <--- Đã sửa từ 'bat_thuong' thành 'anomaly_final'
                    #     color_discrete_map={"Bình thường": "#00CC96", "Bất thường": "#EF553B"},
                    #     hover_data=["quan", "loai_hinh", "dien_tich"],
                    #     labels={
                    #         "prediction": "Giá dự đoán mô hình (VNĐ)",
                    #         "gia_thuc_te": "Giá thực tế (VNĐ)",
                    #         "anomaly_final": "Trạng thái"
                    #     },
                    #     title="So sánh Giá thực tế vs Giá định giá từ AI"
                    # )
                    # # Đường y = x tham chiếu
                    # max_val = float(max(result["prediction"].max(), result["gia_thuc_te"].max()))
                    # fig1.add_shape(
                    #     type="line", line=dict(dash="dash", color="gray"),
                    #     x0=0, x1=max_val, y0=0, y1=max_val
                    # )
                    # fig1.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                    # st.plotly_chart(fig1, use_container_width=True)

                    # Biểu đồ 2: Đơn giá / m2 theo Quận
                    st.write("### 📦 Biểu đồ Phân Bố Đơn Giá (VNĐ/m²) Theo Từng Quận")
                    fig2 = px.box(
                        result,
                        x="quan",
                        y="gia_m2_thuc_te",
                        color="anomaly_final",  
                        color_discrete_map={"Bình thường": "#00CC96", "Bất thường": "#EF553B"},
                        hover_data=["loai_hinh", "dien_tich"],
                        labels={
                            "quan": "Quận / Huyện",
                            "gia_m2_thuc_te": "Đơn giá (VNĐ/m²)",
                            "anomaly_final": "Trạng thái"
                        },
                        title="Khoảng giá chuẩn và điểm Outlier Đơn giá/m² theo Quận"
                    )
                    fig2.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig2, use_container_width=True)

                    st.download_button( "📥 Tải kết quả CSV",
                        data=result.to_csv(index=False).encode("utf-8-sig"),
                        file_name=("ket_qua_phat_hien_bat_thuong.csv"),
                        mime="text/csv",
                        key="p1_download_anomaly"
                        )
        except Exception as e:

            st.error( f"❌ Không thể xử lý file CSV: {e}")





