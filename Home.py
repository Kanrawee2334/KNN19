from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 🎨 การตั้งค่าหน้าเพจและส่วนหัว (Page Setup and Header)
# -----------------------------------------------------

# การตั้งค่าคอนฟิกของหน้าเพจ (ต้องอยู่บนสุด)
st.set_page_config(
    page_title="โปรเจคการจำแนกข้อมูลดอกไม้ Iris 🌸",
    page_icon="🌿",
    layout="wide"  # ใช้ layout แบบกว้าง
)

# ใช้ CSS เพื่อเพิ่มสไตล์ที่กำหนดเอง (Custom CSS)
st.markdown("""
<style>
    /* ซ่อน Streamlit's default header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* สไตล์สำหรับหัวข้อหลัก */
    .stApp > header {
        background-color: #F8F8FF; /* สีฟ้าอ่อนมาก */
        padding: 10px;
        border-bottom: 3px solid #6495ED; /* เส้นขอบด้านล่างสีฟ้า */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* สไตล์สำหรับส่วนเนื้อหาหลัก */
    div.stContainer {
        padding-top: 2rem;
    }
    
    /* สไตล์สำหรับ st.subheader ในส่วนการทำนาย */
    h3 {
        color: #008080; /* สีเขียวอมฟ้า */
    }

</style>
""", unsafe_allow_html=True)


st.title("🌺 โปรเจคการจำแนกข้อมูลดอกไม้ Iris ด้วย KNN")
st.subheader("การจำแนกสายพันธุ์ดอกไม้ Iris โดยใช้โมเดล K-Nearest Neighbors")
st.image("./img/kanrawee.jpg", use_column_width=True) # ใช้เต็มความกว้างของคอลัมน์

# -----------------------------------------------------
# 🖼️ ส่วนแสดงภาพดอกไม้ (Flower Gallery)
# -----------------------------------------------------

st.markdown("---") # เส้นคั่น
st.subheader("🌼 สายพันธุ์ดอกไม้ Iris ที่ใช้ในการจำแนก")
col1, col2, col3 = st.columns(3)

with col1:
    st.image("./img/iris1.jpg", caption="Setosa", use_column_width=True) # เปลี่ยนชื่อหัวข้อเป็น caption
    st.markdown("<p style='text-align: center; color: #DC143C;'>**Setosa**</p>", unsafe_allow_html=True)

with col2:
    st.image("./img/iris2.jpg", caption="Versicolor", use_column_width=True)
    st.markdown("<p style='text-align: center; color: #4682B4;'>**Versicolor**</p>", unsafe_allow_html=True)

with col3:
    st.image("./img/iris3.jpg", caption="Virginica", use_column_width=True)
    st.markdown("<p style='text-align: center; color: #3CB371;'>**Virginica**</p>", unsafe_allow_html=True)

st.markdown("---") # เส้นคั่น

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
    dt = pd.DataFrame() # สร้าง DataFrame ว่างไว้
    
# แสดงข้อมูล 10 แถวแรก
if not dt.empty:
    st.dataframe(dt.head(10), use_container_width=True) # ใช้ st.dataframe แทน st.write และให้ใช้ความกว้างของ container

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
        st.bar_chart(dx2, color='#800080') # เพิ่มสี
    
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
col_input1, col_input2 = st.columns(2)

with col_input1:
    st.subheader("กลีบดอก (Petal)")
    # ใช้ Slider สำหรับค่าที่ควรมีการปรับแต่ง
    pt_len = st.slider("📏 Petal Length (ความยาวกลีบดอก)", min_value=dt['petallength'].min(), max_value=dt['petallength'].max(), value=dt['petallength'].mean(), step=0.1)
    pt_wd = st.slider("↔️ Petal Width (ความกว้างกลีบดอก)", min_value=dt['petalwidth'].min(), max_value=dt['petalwidth'].max(), value=dt['petalwidth'].mean(), step=0.1)

with col_input2:
    st.subheader("กลีบเลี้ยง (Sepal)")
    # ใช้ Number Input สำหรับค่าที่ควรมีการกรอก (หรือจะเปลี่ยนเป็น Slider ก็ได้)
    sp_len = st.number_input("📏 Sepal Length (ความยาวกลีบเลี้ยง)", min_value=dt['sepallength'].min(), max_value=dt['sepallength'].max(), value=dt['sepallength'].mean(), step=0.1, format="%.1f")
    sp_wd = st.number_input("↔️ Sepal Width (ความกว้างกลีบเลี้ยง)", min_value=dt['sepalwidth'].min(), max_value=dt['sepalwidth'].max(), value=dt['sepalwidth'].mean(), step=0.1, format="%.1f")


st.markdown("---") # เส้นคั่น

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
        if out[0] == 'Setosa':
            st.image("./img/iris1.jpg", caption="ผลการทำนาย: Setosa", use_column_width=True)
        elif out[0] == 'Versicolor':
            st.image("./img/iris2.jpg", caption="ผลการทำนาย: Versicolor", use_column_width=True)
        else: # Virginica
            st.image("./img/iris3.jpg", caption="ผลการทำนาย: Virginica", use_column_width=True)
    else:
        st.error("ไม่สามารถทำนายได้ เนื่องจากไม่พบไฟล์ข้อมูล")
else:
    st.info("💡 กรุณาเลือกข้อมูลและกดปุ่ม **ทำนายผล**")
    
# -----------------------------------------------------
# 🚪 ส่วนท้าย (Footer)
# -----------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>จัดทำโดย: [ชื่อผู้จัดทำ/รหัส] | ใช้ Streamlit & Scikit-learn</p>", unsafe_allow_html=True)