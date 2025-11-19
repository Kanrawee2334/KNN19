from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 🎨 การตั้งค่าหน้าเพจและส่วนหัว (Page Setup and Header)
# -----------------------------------------------------

# การตั้งค่าคอนฟิกของหน้าเพจ (ยังคงใช้ layout='wide' แต่เราจะคุมขนาดด้วย CSS)
st.set_page_config(
    page_title="โปรเจคการจำแนกข้อมูลดอกไม้ Iris 🌸",
    page_icon="🌿",
    layout="wide"
)

# ใช้ CSS เพื่อเพิ่มสไตล์ที่กำหนดเอง (Custom CSS)
st.markdown("""
<style>
    /* ซ่อน Streamlit's default header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 🌟🌟 ส่วนที่ถูกเพิ่ม/แก้ไขเพื่อจำกัดความกว้าง 🌟🌟 */
    .block-container {
        /* จำกัดความกว้างสูงสุดของเนื้อหาหลัก */
        max-width: 700px; /* สามารถปรับตัวเลขนี้ได้ (เช่น 600px, 50% หรือ 700px) */
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        margin: auto; /* จัดให้อยู่ตรงกลางจอ */
    }
    
    /* สไตล์สำหรับหัวข้อหลัก */
    .stApp > header {
        background-color: #F8F8FF;
        padding: 10px;
        border-bottom: 3px solid #6495ED;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* สไตล์สำหรับ st.subheader ในส่วนการทำนาย */
    h3 {
        color: #008080;
    }

</style>
""", unsafe_allow_html=True)


st.title("🌺 โปรเจคการจำแนกข้อมูลดอกไม้ Iris ด้วย KNN")
st.subheader("การจำแนกสายพันธุ์ดอกไม้ Iris โดยใช้โมเดล K-Nearest Neighbors")
# *สมมติว่าภาพนี้มีอยู่จริง*
# st.image("./img/kanrawee.jpg", use_column_width=True) 

# --- (ส่วนนี้คือโค้ดเนื้อหาหลักเหมือนเดิม) ---
st.markdown("---")
st.subheader("🌼 สายพันธุ์ดอกไม้ Iris ที่ใช้ในการจำแนก")
col1, col2, col3 = st.columns(3)

with col1:
    # *สมมติว่าภาพนี้มีอยู่จริง*
    # st.image("./img/iris1.jpg", caption="Setosa", use_column_width=True)
    st.markdown("<p style='text-align: center; color: #DC143C;'>**Setosa**</p>", unsafe_allow_html=True)

with col2:
    # *สมมติว่าภาพนี้มีอยู่จริง*
    # st.image("./img/iris2.jpg", caption="Versicolor", use_column_width=True)
    st.markdown("<p style='text-align: center; color: #4682B4;'>**Versicolor**</p>", unsafe_allow_html=True)

with col3:
    # *สมมติว่าภาพนี้มีอยู่จริง*
    # st.image("./img/iris3.jpg", caption="Virginica", use_column_width=True)
    st.markdown("<p style='text-align: center; color: #3CB371;'>**Virginica**</p>", unsafe_allow_html=True)

st.markdown("---") 

# -----------------------------------------------------
# 📊 ส่วนแสดงข้อมูลและสถิติ (Data and Statistics)
# -----------------------------------------------------

html_7 = """
<div style="background-color:#F5B7B1; padding:15px; border-radius:15px; border:1px solid #E9967A; color:black;">
    <center><h3>📖 สถิติข้อมูลดอกไม้ Iris Dataset</h3></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")

# โหลดข้อมูล
try:
    dt = pd.read_csv("./data/iris (1).csv")
except FileNotFoundError:
    st.error("ไม่พบไฟล์ข้อมูล './data/iris (1).csv'")
    dt = pd.DataFrame({'petallength':[1.4, 1.4, 1.3], 'petalwidth':[0.2, 0.2, 0.2], 'sepallength':[5.1, 4.9, 4.7], 'sepalwidth':[3.5, 3.0, 3.2], 'variety':['Setosa', 'Setosa', 'Setosa']})

# แสดงข้อมูล 10 แถวแรก
if not dt.empty:
    st.dataframe(dt.head(10), use_container_width=True) 

    # คำนวณผลรวม
    dt1 = dt['petallength'].sum()
    dt2 = dt['petalwidth'].sum()
    dt3 = dt['sepallength'].sum()
    dt4 = dt['sepalwidth'].sum()

    dx = [dt1, dt2, dt3, dt4]
    dx2 = pd.DataFrame(dx, index=["Petal Length Sum", "Petal Width Sum", "Sepal Length Sum", "Sepal Width Sum"])
    
    st.markdown("---") 

    # สร้าง Container สำหรับ Chart
    with st.expander("📈 แสดงภาพรวมผลรวมข้อมูล (Data Visualization)"):
        st.bar_chart(dx2, color='#800080')
    
    st.markdown("---") 

# -----------------------------------------------------
# 🧠 ส่วนทำนายผล (Prediction Section)
# -----------------------------------------------------

html_8 = """
<div style="background-color:#F7DC6F; padding:15px; border-radius:15px; border:1px solid #DAA520; color:black;">
    <center><h3>🔮 ทำนายสายพันธุ์ดอกไม้ด้วย KNN</h3></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

# สร้าง Column สำหรับการรับค่า
# ใช้ค่าเฉลี่ยเป็นค่าเริ่มต้นเพื่อป้องกัน error หากไฟล์ข้อมูลไม่เจอ
if not dt.empty:
    default_pl = dt['petallength'].mean()
    default_pw = dt['petalwidth'].mean()
    default_sl = dt['sepallength'].mean()
    default_sw = dt['sepalwidth'].mean()
    min_pl, max_pl = dt['petallength'].min(), dt['petallength'].max()
    min_pw, max_pw = dt['petalwidth'].min(), dt['petalwidth'].max()
    min_sl, max_sl = dt['sepallength'].min(), dt['sepallength'].max()
    min_sw, max_sw = dt['sepalwidth'].min(), dt['sepalwidth'].max()
else:
    default_pl, default_pw, default_sl, default_sw = 1.4, 0.2, 5.1, 3.5
    min_pl, max_pl = 0.1, 6.9
    min_pw, max_pw = 0.1, 2.5
    min_sl, max_sl = 4.3, 7.9
    min_sw, max_sw = 2.0, 4.4


col_input1, col_input2 = st.columns(2)

with col_input1:
    st.subheader("กลีบดอก (Petal)")
    pt_len = st.slider("📏 Petal Length (ความยาวกลีบดอก)", min_value=min_pl, max_value=max_pl, value=default_pl, step=0.1)
    pt_wd = st.slider("↔️ Petal Width (ความกว้างกลีบดอก)", min_value=min_pw, max_value=max_pw, value=default_pw, step=0.1)

with col_input2:
    st.subheader("กลีบเลี้ยง (Sepal)")
    sp_len = st.number_input("📏 Sepal Length (ความยาวกลีบเลี้ยง)", min_value=min_sl, max_value=max_sl, value=default_sl, step=0.1, format="%.1f")
    sp_wd = st.number_input("↔️ Sepal Width (ความกว้างกลีบเลี้ยง)", min_value=min_sw, max_value=max_sw, value=default_sw, step=0.1, format="%.1f")


st.markdown("---")

# ปุ่มทำนาย
if st.button("🚀 ทำนายผล", use_container_width=True, help="คลิกเพื่อทำการจำแนกสายพันธุ์ดอกไม้"):
    if not dt.empty:
        # เตรียมข้อมูลสำหรับ KNN Model
        X = dt.drop('variety', axis=1)
        y = dt.variety
        
        # ฝึกโมเดล KNN
        Knn_model = KNeighborsClassifier(n_neighbors=3)
        Knn_model.fit(X, y)
        
        # เตรียมข้อมูลอินพุต
        x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
        
        # ทำนายผล
        out = Knn_model.predict(x_input)
        
        # แสดงผลลัพธ์
        st.success(f"✅ ผลการทำนาย: สายพันธุ์ **{out[0]}**")
        
        # แสดงภาพตามผลการทำนาย
        # *สมมติว่าภาพเหล่านี้มีอยู่จริง*
        # if out[0] == 'Setosa':
        #     st.image("./img/iris1.jpg", caption="ผลการทำนาย: Setosa", use_column_width=True)
        # elif out[0] == 'Versicolor':
        #     st.image("./img/iris2.jpg", caption="ผลการทำนาย: Versicolor", use_column_width=True)
        # else: # Virginica
        #     st.image("./img/iris3.jpg", caption="ผลการทำนาย: Virginica", use_column_width=True)
        st.markdown(f"**ภาพจำลองสำหรับสายพันธุ์ {out[0]}**")
    else:
        st.error("ไม่สามารถทำนายได้ เนื่องจากไม่พบไฟล์ข้อมูล")
else:
    st.info("💡 กรุณาเลือกข้อมูลและกดปุ่ม **ทำนายผล**")
    
# -----------------------------------------------------
# 🚪 ส่วนท้าย (Footer)
# -----------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>จัดทำโดย: [ชื่อผู้จัดทำ/รหัส] | ใช้ Streamlit & Scikit-learn</p>", unsafe_allow_html=True)