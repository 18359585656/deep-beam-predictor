# ==========================================
# app.py
# 深受弯构件受剪承载力智能预测系统
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# 页面配置
# ==========================================
st.set_page_config(
    page_title="深受弯构件承载力预测系统",
    page_icon="🧱",
    layout="wide"
)

# ==========================================
# 当前目录
# ==========================================
BASE_DIR = os.path.dirname(__file__)

# ==========================================
# 加载模型
# ==========================================
@st.cache_resource
def load_models():

    models = {}

    # ------------------------------
    # 实腹模型
    # ------------------------------
    try:

        solid_model_path = os.path.join(
            BASE_DIR,
            "solid_model.pkl"
        )

        solid_cols_path = os.path.join(
            BASE_DIR,
            "solid_columns.pkl"
        )

        models["solid_model"] = joblib.load(
            solid_model_path
        )

        models["solid_cols"] = joblib.load(
            solid_cols_path
        )

    except Exception as e:

        st.error(f"❌ 实腹模型加载失败：{e}")

        models["solid_model"] = None
        models["solid_cols"] = None

    # ------------------------------
    # 开洞模型
    # ------------------------------
    try:

        opening_model_path = os.path.join(
            BASE_DIR,
            "opening_model.pkl"
        )

        opening_cols_path = os.path.join(
            BASE_DIR,
            "opening_columns.pkl"
        )

        models["opening_model"] = joblib.load(
            opening_model_path
        )

        models["opening_cols"] = joblib.load(
            opening_cols_path
        )

    except Exception as e:

        st.error(f"❌ 开洞模型加载失败：{e}")

        models["opening_model"] = None
        models["opening_cols"] = None

    return models


models = load_models()

# ==========================================
# 调试信息
# ==========================================
with st.expander("🔍 调试信息", expanded=False):

    st.write("当前目录：")
    st.code(BASE_DIR)

    st.write("当前目录文件：")
    st.write(os.listdir(BASE_DIR))

# ==========================================
# 页面标题
# ==========================================
st.title("🧱 深受弯构件受剪承载力智能预测系统")

st.markdown("""
基于机器学习算法构建的深受弯构件受剪承载力预测平台
""")

st.divider()

# ==========================================
# 侧边栏
# ==========================================
st.sidebar.header("🛠️ 参数输入")

beam_type = st.sidebar.radio(
    "构件类型",
    [
        "实腹深受弯构件",
        "开洞深受弯构件"
    ]
)

# ==========================================
# 实腹深受弯构件
# ==========================================
if beam_type == "实腹深受弯构件":

    st.sidebar.subheader("几何与材料")

    b = st.sidebar.number_input(
        "截面宽度 b (mm)",
        value=200.0
    )

    h = st.sidebar.number_input(
        "截面高度 h (mm)",
        value=600.0
    )

    a_h = st.sidebar.number_input(
        "剪跨比 a/h",
        value=1.0,
        min_value=0.2,
        max_value=2.5,
        step=0.01
    )

    fc = st.sidebar.number_input(
        "混凝土强度 fc (MPa)",
        value=30.0
    )

    aggregate_type = st.sidebar.selectbox(
        "混凝土类型",
        [
            "普通混凝土",
            "轻骨料混凝土"
        ]
    )

    aggregate_val = 1 if aggregate_type == "普通混凝土" else 2

    st.sidebar.subheader("配筋参数")

    pl = st.sidebar.number_input(
        "纵筋配筋率 pl (%)",
        value=1.2
    )

    fy = st.sidebar.number_input(
        "纵筋屈服强度 fy (MPa)",
        value=400.0
    )

    pv = st.sidebar.number_input(
        "箍筋配筋率 pv (%)",
        value=0.5
    )

    fyv = st.sidebar.number_input(
        "箍筋屈服强度 fyv (MPa)",
        value=300.0
    )

    ph = st.sidebar.number_input(
        "水平筋配筋率 ph (%)",
        value=0.5
    )

    fyh = st.sidebar.number_input(
        "水平筋屈服强度 fyh (MPa)",
        value=300.0
    )

    input_dict = {
        "b": b,
        "h": h,
        "a/h": a_h,
        "fc": fc,
        "pl": pl,
        "fy": fy,
        "ph": ph,
        "fyh": fyh,
        "pv": pv,
        "fyv": fyv,
        "Aggregate": aggregate_val
    }

    model = models["solid_model"]

    model_cols = models["solid_cols"]

# ==========================================
# 开洞深受弯构件
# ==========================================
else:

    st.sidebar.subheader("几何与材料")

    b = st.sidebar.number_input(
        "构件宽度 b (mm)",
        value=200.0
    )

    a_h = st.sidebar.number_input(
        "剪跨比 a/h",
        value=1.0,
        min_value=0.2,
        max_value=2.5,
        step=0.01
    )

    fc = st.sidebar.number_input(
        "混凝土强度 fc (MPa)",
        value=30.0
    )

    st.sidebar.subheader("开洞参数")

    m1 = st.sidebar.number_input(
        "m1",
        value=0.30
    )

    m2 = st.sidebar.number_input(
        "m2",
        value=0.50
    )

    k1 = st.sidebar.number_input(
        "k1",
        value=0.20
    )

    k2 = st.sidebar.number_input(
        "k2",
        value=0.20
    )

    st.sidebar.subheader("配筋参数")

    plfy = st.sidebar.number_input(
        "plfy",
        value=8.0
    )

    phfyh = st.sidebar.number_input(
        "phfyh",
        value=1.5
    )

    input_dict = {
        "b": b,
        "a/h": a_h,
        "m1": m1,
        "m2": m2,
        "k1": k1,
        "k2": k2,
        "plfy": plfy,
        "phfyh": phfyh,
        "fc": fc
    }

    model = models["opening_model"]

    model_cols = models["opening_cols"]

# ==========================================
# 模型检查
# ==========================================
if model is None or model_cols is None:

    st.error("❌ 模型文件未正确加载")

# ==========================================
# 开始预测
# ==========================================
else:

    input_df = pd.DataFrame([input_dict])

    final_input = pd.DataFrame()

    missing_cols = []

    # 保持训练时列顺序
    for col in model_cols:

        if col in input_df.columns:

            final_input[col] = input_df[col]

        else:

            final_input[col] = 0.0

            missing_cols.append(col)

    # 页面布局
    col1, col2 = st.columns([1, 1.5])

    # 左侧
    with col1:

        st.info("### 📝 当前输入参数")

        st.dataframe(
            input_df,
            use_container_width=True
        )

        calc_btn = st.button(
            "🚀 计算承载力",
            type="primary",
            use_container_width=True
        )

    # 右侧
    with col2:

        if calc_btn:

            try:

                pred = model.predict(final_input)[0]

                st.success("✅ 计算完成")

                st.markdown("## 预测极限受剪承载力")

                st.markdown(
                    f"""
                    <h1 style='color:#2e7d32;'>
                    {pred:.2f} kN
                    </h1>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander("查看模型输入"):

                    st.dataframe(
                        final_input,
                        use_container_width=True
                    )

                    if missing_cols:

                        st.warning(
                            f"以下字段缺失，已自动补0：{missing_cols}"
                        )

            except Exception as e:

                st.error(f"❌ 计算失败：{e}")

        else:

            st.info("👈 请点击按钮开始计算")

# ==========================================
# 水印
# ==========================================
st.markdown(
    """
    <style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        padding: 5px 10px;
        background-color: rgba(255,255,255,0.7);
        color: #888888;
        font-size: 14px;
        border-radius: 5px;
        z-index: 9999;
        pointer-events: none;
    }
    </style>

    <div class="watermark">
        © 2025 Developed by Li Yuanxi
    </div>
    """,
    unsafe_allow_html=True
)
