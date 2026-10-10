import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Dashboard Phân Cụm & Học Máy", layout="wide")

# 1. Load dữ liệu an toàn
@st.cache_data
def load_data():
    try:
        path = r'C:\Users\Mr Huy\Downloads\Bài nhóm_final\DHHTTT20B_TCTA_Nhom8_Stage1\data\merged_data_clean.csv'
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"Không thể load file: {e}")
        df = pd.DataFrame()
    return df

df = load_data()

if not df.empty:
    # Ép tên các cột trong DataFrame về chữ thường để tránh lỗi lệch ký tự hoa/thường
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Sidebar chọn tính năng / mô hình
    st.sidebar.title("🧭 Điều hướng mô hình")
    app_mode = st.sidebar.selectbox("Chọn chức năng:", [
        "PCA + K-Means (Phân cụm)", 
        "Linear Regression (Hồi quy tuyến tính)", 
        "Decision Tree Regressor", 
        "Logistic Regression (Phân loại)", 
        "Decision Tree Classifier"
    ])

    # Các cột yêu cầu theo bảng của bạn
    feature_cols = ['price', 'rating', 'views', 'add_to_cart']
    target_col = 'purchase'
    
    # Kiểm tra xem các cột cần thiết có tồn tại trong df không
    missing_cols = [col for col in ['price', 'rating', 'views', 'add_to_cart', 'purchase'] if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Dữ liệu thiếu các cột sau: {missing_cols}. Hãy kiểm tra lại file CSV.")

    # --- 1. PCA + K-Means ---
    if app_mode == "PCA + K-Means (Phân cụm)":
        st.title("📊 Mô hình: PCA + K-Means Clustering")
        st.markdown("Giảm chiều dữ liệu bằng PCA và phân nhóm cụm khách hàng/sản phẩm.")
        
        k_clusters = st.sidebar.slider("Chọn số lượng cụm (K):", 2, 8, 3)
        
        req_cols = ['price', 'rating', 'views', 'add_to_cart', 'purchase']
        if all(col in df.columns for col in req_cols):
            X = df[req_cols].fillna(0)
            
            # 1. Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 2. PCA giảm xuống 2 chiều
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # 3. K-Means
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(X_pca)
            
            st.subheader("Kết quả phân cụm mẫu:")
            display_cols = ['product_name', 'price', 'rating', 'Cluster'] if 'product_name' in df.columns else ['price', 'rating', 'Cluster']
            st.dataframe(df[display_cols].head(10))
            
            # Hiển thị biểu đồ phân tán các cụm
            chart_data = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            chart_data['Cluster'] = df['Cluster'].astype(str) # Ép kiểu string để hiện màu
            if 'product_name' in df.columns:
                chart_data['product_name'] = df['product_name']
            
            st.subheader("Biểu đồ phân cụm PCA (2D)")
            st.scatter_chart(chart_data, x='PC1', y='PC2', color='Cluster')

    # --- 2. Linear Regression ---
    elif app_mode == "Linear Regression (Hồi quy tuyến tính)":
        st.title("📈 Mô hình: Linear Regression")
        st.markdown("Dự đoán **Số lượt mua (purchase)** dựa trên các đặc trưng đầu vào.")
            
        if all(col in df.columns for col in feature_cols) and target_col in df.columns:
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(0)
                
            # Huấn luyện mô hình
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
                
            score = model.score(X_test, y_test)
            st.metric(label="Độ chính xác R² Score trên tập Test", value=f"{score:.2f}")
                
            st.markdown("---")
            st.subheader("🔮 Dự đoán thử nghiệm sản phẩm mới:")
                
            col_input1, col_input2 = st.columns(2)
            with col_input1:
                p = st.number_input("Price (Giá bán - VNĐ)", value=150000.0, key="lr_p")
                r = st.slider("Rating (Đánh giá)", 1.0, 5.0, 4.5, key="lr_r")
            with col_input2:
                v = st.number_input("Views (Lượt xem)", value=1300, key="lr_v")
                atc = st.number_input("Add to Cart (Thêm vào giỏ)", value=170, key="lr_atc")
                
            if st.button("Thực hiện dự đoán số lượt mua", key="btn_lr"):
                pred = model.predict([[p, r, v, atc]])[0]
                final_pred = max(0, int(round(pred)))
                st.success(f"📌 Số lượt mua dự đoán: **{final_pred} lượt**")
        else:
            st.error("❌ Thiếu dữ liệu cột đặc trưng hoặc cột target.")

    # --- 3. Decision Tree Regressor ---
    elif app_mode == "Decision Tree Regressor":
        st.title("🌳 Mô hình: Decision Tree Regressor")
        st.markdown("Dự đoán **Số lượt mua (purchase)** bằng cây quyết định.")
        
        if all(col in df.columns for col in feature_cols) and target_col in df.columns:
            X = df[feature_cols].fillna(0)
            y = df[target_col].fillna(0)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X_train, y_train)
            
            score = model.score(X_test, y_test)
            st.metric(label="Độ chính xác R² Score trên tập Test", value=f"{score:.2f}")
            
            st.markdown("---")
            st.subheader("🔮 Dự đoán thử nghiệm sản phẩm mới:")
            
            col_input1, col_input2 = st.columns(2)
            with col_input1:
                p = st.number_input("Price (Giá bán - VNĐ)", value=150000.0, key="dtr_p")
                r = st.slider("Rating (Đánh giá)", 1.0, 5.0, 4.5, key="dtr_r")
            with col_input2:
                v = st.number_input("Views (Lượt xem)", value=1300, key="dtr_v")
                atc = st.number_input("Add to Cart (Thêm vào giỏ)", value=170, key="dtr_atc")
                
            if st.button("Thực hiện dự đoán số lượt mua", key="btn_dtr"):
                pred = model.predict([[p, r, v, atc]])[0]
                final_pred = max(0, int(round(pred)))
                st.success(f"📌 Số lượt mua dự đoán (Decision Tree): **{final_pred} lượt**")
        else:
            st.error("❌ Thiếu dữ liệu cột đặc trưng hoặc cột target.")

    # --- 4. Logistic Regression ---
    elif app_mode == "Logistic Regression (Phân loại)":
        st.title("⚖️ Mô hình: Logistic Regression")
        st.markdown("Phân loại: Khách hàng **Có mua (1)** hay **Không mua (0)**.")
        
        if all(col in df.columns for col in feature_cols) and target_col in df.columns:
            df['target'] = (df[target_col] > 0).astype(int)
            X = df[feature_cols].fillna(0)
            y = df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
            score = model.score(X_test, y_test)
            st.metric(label="Độ chính xác (Accuracy) trên tập Test", value=f"{score * 100:.2f}%")
            
            st.markdown("---")
            st.subheader("🔮 Dự đoán khả năng mua của khách hàng:")
            
            col_input1, col_input2 = st.columns(2)
            with col_input1:
                p = st.number_input("Price (Giá bán - VNĐ)", value=150000.0, key="log_p")
                r = st.slider("Rating (Đánh giá)", 1.0, 5.0, 4.5, key="log_r")
            with col_input2:
                v = st.number_input("Views (Lượt xem)", value=1300, key="log_v")
                atc = st.number_input("Add to Cart (Thêm vào giỏ)", value=170, key="log_atc")
                
            if st.button("Thực hiện phân loại", key="btn_log"):
                pred = model.predict([[p, r, v, atc]])[0]
                result_text = "Có mua (1)" if pred == 1 else "Không mua (0)"
                st.success(f"📌 Kết quả phân loại: **{result_text}**")
        else:
            st.error("❌ Thiếu dữ liệu cột đặc trưng hoặc cột target.")

    # --- 5. Decision Tree Classifier ---
    elif app_mode == "Decision Tree Classifier":
        st.title("🌲 Mô hình: Decision Tree Classifier")
        st.markdown("Phân loại khách hàng **Có mua (1)** / **Không mua (0)** bằng cây quyết định.")
        
        if all(col in df.columns for col in feature_cols) and target_col in df.columns:
            df['target'] = (df[target_col] > 0).astype(int)
            X = df[feature_cols].fillna(0)
            y = df['target']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            score = model.score(X_test, y_test)
            st.metric(label="Độ chính xác (Accuracy) trên tập Test", value=f"{score * 100:.2f}%")
            
            st.markdown("---")
            st.subheader("🔮 Dự đoán khả năng mua của khách hàng:")
            
            col_input1, col_input2 = st.columns(2)
            with col_input1:
                p = st.number_input("Price (Giá bán - VNĐ)", value=150000.0, key="dtc_p")
                r = st.slider("Rating (Đánh giá)", 1.0, 5.0, 4.5, key="dtc_r")
            with col_input2:
                v = st.number_input("Views (Lượt xem)", value=1300, key="dtc_v")
                atc = st.number_input("Add to Cart (Thêm vào giỏ)", value=170, key="dtc_atc")
                
            if st.button("Thực hiện phân loại", key="btn_dtc"):
                pred = model.predict([[p, r, v, atc]])[0]
                result_text = "Có mua (1)" if pred == 1 else "Không mua (0)"
                st.success(f"📌 Kết quả phân loại (Decision Tree): **{result_text}**")
        else:
            st.error("❌ Thiếu dữ liệu cột đặc trưng hoặc cột target.")

else:
    st.warning("⚠️ Không có dữ liệu để hiển thị giao diện. Vui lòng kiểm tra lại đường dẫn file CSV.")