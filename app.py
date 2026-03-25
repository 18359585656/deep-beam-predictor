# ==========================================
# app.py: 深受弯构件智能预测系统（实腹 + 开洞）
# ==========================================
import streamlit as st
import pandas as pd
import joblib

# 页面设置
st.set_page_config(page_title="深受弯构件承载力预测系统", layout="wide", page_icon="🧱")

# --- 1. 加载模型 ---
@st.cache_resource
def load_models():
    result = {}

    try:
        result["solid_model"] = joblib.load("solid_model.pkl")
        result["solid_cols"] = joblib.load("solid_columns.pkl")
    except:
        result["solid_model"] = None
        result["solid_cols"] = None

    try:
        result["opening_model"] = joblib.load("opening_model.pkl")
        result["opening_cols"] = joblib.load("opening_columns.pkl")
    except:
        result["opening_model"] = None
        result["opening_cols"] = None

    return result

models = load_models()

# --- 2. 侧边栏：参数输入 ---
st.sidebar.header("🛠️ 设计参数输入")

beam_type = st.sidebar.radio(
    "构件类型选择",
    ["实腹深受弯构件", "开洞深受弯构件"]
)

# ==========================================
# A. 实腹深受弯构件输入
# ==========================================
if beam_type == "实腹深受弯构件":
    with st.sidebar.expander("1. 几何与材料", expanded=True):
        b = st.number_input("截面宽度 $b$ (mm)", value=200.0, step=10.0)
        h = st.number_input("截面高度 $h$ (mm)", value=600.0, step=10.0)
        a_h = st.number_input("剪跨比 $a/h$", min_value=0.2, max_value=2.5, value=1.0, step=0.01, format="%.2f")

        agg_option = st.radio(
            "混凝土/骨料类型 (Aggregate)",
            ("普通混凝土 (Normal)", "轻骨料混凝土 (Lightweight)"),
            index=0
        )
        aggregate_val = 1 if "普通" in agg_option else 2

        fc = st.number_input("混凝土强度 $f_c$ (MPa)", value=30.0, step=5.0)

    st.sidebar.subheader("2. 配筋信息")

    st.sidebar.markdown("##### 🟢 纵向钢筋")
    pl = st.sidebar.number_input("配筋率 $\\rho_l$ (%)", value=1.2, step=0.1, format="%.2f")
    fy = st.sidebar.number_input("纵筋屈服强度 $f_y$ (MPa)", value=400.0, step=10.0, format="%.1f")

    st.sidebar.markdown("##### 🔵 竖向腹筋 (箍筋)")
    pv = st.sidebar.number_input("配筋率 $\\rho_v$ (%)", value=0.5, step=0.1, format="%.2f")
    fyv = st.sidebar.number_input("箍筋屈服强度 $f_{yv}$ (MPa)", value=300.0, step=10.0, format="%.1f")

    st.sidebar.markdown("##### 🟠 水平腹筋")
    ph = st.sidebar.number_input("配筋率 $\\rho_h$ (%)", value=0.5, step=0.1, format="%.2f")
    fyh = st.sidebar.number_input("水平筋屈服强度 $f_{yh}$ (MPa)", value=300.0, step=10.0, format="%.1f")

    input_dict = {
        'b': b,
        'h': h,
        'a/h': a_h,
        'fc': fc,
        'pl': pl,
        'fy': fy,
        'ph': ph,
        'fyh': fyh,
        'pv': pv,
        'fyv': fyv,
        'Aggregate': int(aggregate_val)
    }

    model = models["solid_model"]
    model_cols = models["solid_cols"]

# ==========================================
# B. 开洞深受弯构件输入
# ==========================================
else:
    with st.sidebar.expander("1. 几何与材料", expanded=True):
        b = st.number_input("构件宽度 $b$ (mm)", value=200.0, step=10.0)
        a_h = st.number_input("剪跨比 $a/h$", min_value=0.2, max_value=2.5, value=1.0, step=0.01, format="%.2f")
        fc = st.number_input("混凝土强度 $f_c$ (MPa)", value=30.0, step=5.0)

    with st.sidebar.expander("2. 开洞参数", expanded=True):
        m1 = st.number_input("开洞位置参数 $m_1$", value=0.30, step=0.01, format="%.2f")
        m2 = st.number_input("开洞位置参数 $m_2$", value=0.50, step=0.01, format="%.2f")
        k1 = st.number_input("开洞尺寸参数 $k_1$", value=0.20, step=0.01, format="%.2f")
        k2 = st.number_input("开洞尺寸参数 $k_2$", value=0.20, step=0.01, format="%.2f")

    with st.sidebar.expander("3. 配筋参数", expanded=True):
        plfy = st.number_input("纵向配筋特征参数 $plfy$", value=8.0, step=0.1, format="%.2f")
        phfyh = st.number_input("水平配筋特征参数 $phfyh$", value=1.5, step=0.1, format="%.2f")

    input_dict = {
        'b': b,
        'a/h': a_h,
        'm1': m1,
        'm2': m2,
        'k1': k1,
        'k2': k2,
        'plfy': plfy,
        'phfyh': phfyh,
        'fc': fc
    }

    model = models["opening_model"]
    model_cols = models["opening_cols"]

# --- 3. 主界面展示 ---
if model is None or model_cols is None:
    st.error("❌ 模型文件丢失！请确保对应的 pkl 文件在当前目录下。")
else:
    input_df = pd.DataFrame([input_dict])
    final_input = pd.DataFrame()

    # 按训练时的列顺序重排
    missing_cols = []
    for col in model_cols:
        if col in input_df.columns:
            final_input[col] = input_df[col]
        else:
            final_input[col] = 0.0
            missing_cols.append(col)

    if missing_cols:
        print(f"警告：模型需要以下列，但输入中未找到（已自动填0）：{missing_cols}")

    st.title("🧱 深受弯构件受剪承载力智能预测系统")
    st.markdown("基于 **机器学习算法** 开发")
    st.divider()

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.info("### 📝 当前参数概览")

        if beam_type == "实腹深受弯构件":
            type_display = "普通混凝土 (Normal)" if aggregate_val == 1 else "轻骨料混凝土 (Lightweight)"

            st.markdown(f"""
            * **构件类型**: **实腹深受弯构件**
            * **材料类型**: **{type_display}**
            * **截面尺寸**: ${b:.0f} \\times {h:.0f}$ mm （剪跨比 $a/h={a_h:.2f}$）
            * **混凝土**: $f_c = {fc:.1f}$ MPa
            * **🟢 纵筋**: $\\rho_l = {pl:.2f}\\%$ （$f_y={fy:.0f}$ MPa）
            * **🔵 箍筋**: $\\rho_v = {pv:.2f}\\%$ （$f_{{yv}}={fyv:.0f}$ MPa）
            * **🟠 水平筋**: $\\rho_h = {ph:.2f}\\%$ （$f_{{yh}}={fyh:.0f}$ MPa）
            """)
        else:
            st.markdown(f"""
            * **构件类型**: **开洞深受弯构件**
            * **构件宽度**: $b = {b:.0f}$ mm
            * **剪跨比**: $a/h = {a_h:.2f}$
            * **混凝土**: $f_c = {fc:.1f}$ MPa
            * **开洞位置参数**: $m_1 = {m1:.2f}$，$m_2 = {m2:.2f}$
            * **开洞尺寸参数**: $k_1 = {k1:.2f}$，$k_2 = {k2:.2f}$
            * **纵向配筋特征参数**: $plfy = {plfy:.2f}$
            * **水平配筋特征参数**: $phfyh = {phfyh:.2f}$
            """)

        calc_btn = st.button("🚀 计算承载力", type="primary", use_container_width=True)

    with col2:
        if calc_btn:
            try:
                pred = model.predict(final_input)[0]

                st.success("### ✅ 计算完成")
                st.markdown("##### 预测极限受剪承载力 $V_u$")
                st.markdown(
                    f"<h1 style='text-align: left; color: #2e7d32;'>{pred:.2f} kN</h1>",
                    unsafe_allow_html=True
                )

                with st.expander("查看详细数据"):
                    st.write("输入模型的特征矩阵：")
                    st.dataframe(final_input, use_container_width=True)

                    if missing_cols:
                        st.warning(f"以下列未在输入中提供，已自动补 0：{missing_cols}")

            except Exception as e:
                st.error(f"计算出错: {str(e)}")
                st.warning("请检查输入字段名是否与模型训练时的特征名完全一致。")
        else:
            st.write("👈 请在左侧调整参数并点击计算")

# --- 4. 个性化水印 ---
st.markdown("""
    <style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        width: auto;
        padding: 5px 10px;
        background-color: rgba(255, 255, 255, 0.7);
        color: #888888;
        font-size: 14px;
        border-radius: 5px;
        z-index: 9999;
        pointer-events: none;
        font-family: sans-serif;
    }
    @media (prefers-color-scheme: dark) {
        .watermark {
            background-color: rgba(40, 40, 40, 0.7);
            color: #bbbbbb;
        }
    }
    </style>

    <div class="watermark">
        © 2025 Developed by Li Yuanxi (Chang'an University) | 毕业设计专用
    </div>
""", unsafe_allow_html=True)