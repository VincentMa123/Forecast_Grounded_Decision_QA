import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

'''
使用方式:
streamlit run data/vis_parquet_sequence.py
'''

# 设置页面配置
st.set_page_config(
    page_title="序列数据可视化",
    page_icon="📊",
    layout="wide"
)

st.title("📊 序列数据可视化工具")
st.markdown("查看原始数据中变量的时间序列")

# 侧边栏配置
st.sidebar.header("📋 选择配置")

# 1. 选择数据源
data_source = st.sidebar.selectbox(
    "数据源",
    ["train_sequences.parquet", "val_sequences.parquet"]
)

# 加载变量名映射
@st.cache_data
def load_variable_mapping(mapping_path):
    """加载变量索引映射"""
    if not mapping_path.exists():
        return None
    return pd.read_csv(mapping_path)

# 加载数据
@st.cache_data
def load_data(file_path):
    """加载parquet文件并反序列化数据"""
    if not file_path.exists():
        return None
    df = pd.read_parquet(file_path)
    return df

cache_dir = Path("./data/cache_15pct")
mapping_file = Path(__file__).resolve().parent / "index_variable_mapping.csv"
default_static_mapping = Path(__file__).resolve().parent / "static" / "full" / "index_variable_mapping.csv"
if default_static_mapping.exists():
    mapping_file = default_static_mapping

# 加载变量映射
var_mapping = load_variable_mapping(mapping_file)
if var_mapping is None:
    st.error(f"❌ 变量映射文件不存在: {mapping_file}")
    st.stop()

parquet_file = cache_dir / data_source
df = load_data(parquet_file)

if df is None:
    st.error(f"❌ 文件不存在: {parquet_file}")
    st.stop()

# 显示基本信息
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 数据信息")

# 新格式：原始样本数据
st.sidebar.write(f"样本数: {len(df)}")

# 获取数据形状
first_row = df.iloc[0]
data_shape = (first_row['shape_0'], first_row['shape_1'])  # [1439, 6712]
st.sidebar.write(f"数据形状: {data_shape}")
st.sidebar.write(f"时间步/样本: {data_shape[0]}")
st.sidebar.write(f"变量数: {data_shape[1]}")

# 2. 选择样本ID
st.sidebar.markdown("---")
sample_ids = df['sample_id'].unique()
selected_sample = st.sidebar.selectbox(
    "选择样本ID",
    options=sample_ids,
    index=0
)

# 从原始样本加载数据
selected_sample_row = df[df['sample_id'] == selected_sample].iloc[0]

# 反序列化原始数据
raw_data = np.frombuffer(selected_sample_row['data'], dtype=np.float32).reshape(
    (selected_sample_row['shape_0'], selected_sample_row['shape_1'])
)

# 3. 选择时间窗口
st.sidebar.markdown("---")
st.sidebar.markdown("### 🕐 时间窗口选择")

window_start = st.sidebar.slider(
    "时间窗口起始位置",
    min_value=0,
    max_value=data_shape[0] - 1,
    value=0,
    step=10
)

window_size = st.sidebar.slider(
    "时间窗口大小",
    min_value=10,
    max_value=min(data_shape[0] - window_start, 200),
    value=60,
    step=10
)

st.sidebar.write(f"查看范围: {window_start} ~ {window_start + window_size}")

# 准备时间窗口数据
input_data = raw_data[window_start:window_start + window_size]
target_data = raw_data[window_start + 1:window_start + 1 + window_size]

# 如果target超出范围，截断
if len(target_data) > len(input_data):
    target_data = target_data[:len(input_data)]

time_steps = list(range(len(input_data)))

# 4. 选择变量（显示变量名）
st.sidebar.markdown("---")
var_options = [f"{row['index']}: {row['variable_name']}" for _, row in var_mapping.iterrows()]
var_selection = st.sidebar.selectbox(
    "选择变量",
    options=var_options,
    index=0
)
var_idx = int(var_selection.split(":")[0])
var_name = var_mapping.loc[var_mapping['index'] == var_idx, 'variable_name'].values[0]

# 主界面显示
st.subheader(f"📊 样本 {selected_sample} - 变量 {var_name}")

# 添加格式说明
st.info("🔍 **直接可视化**: 从原始样本数据中选择时间窗口和变量进行查看")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("样本ID", selected_sample)
with col2:
    st.metric("时间窗口", f"{window_start}-{window_start + window_size}")
with col3:
    st.metric("时间步数", len(input_data))
with col4:
    # 显示样本的时间范围
    start_time = selected_sample_row['start_time']
    end_time = selected_sample_row['end_time']
    st.metric("样本时间", f"{start_time} → {end_time}")

st.markdown("---")

# 显示图表和数值
tab1, tab2, tab3 = st.tabs(["📈 时间序列图", "📋 数值列表", "📊 统计信息"])

with tab1:
    st.subheader(f"变量 {var_name} (索引: {var_idx}) 的时间序列")

    # 对比图 - 显示input和target的折线图
    st.markdown("### Input vs Target 对比图")
    fig = go.Figure()

    # Input data
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=input_data[:, var_idx],
        mode='lines+markers',
        name='Input',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))

    # Target data
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=target_data[:, var_idx],
        mode='lines+markers',
        name='Target (t+1)',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title=f"样本 {selected_sample} - {var_name} 时间窗口 {window_start}-{window_start + window_size}",
        xaxis_title="时间步",
        yaxis_title="数值",
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    st.plotly_chart(fig, width='stretch')

    # 分离的对比图
    st.markdown("### 分离显示")
    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"Input Data - {var_name}", f"Target Data - {var_name}"),
        vertical_spacing=0.12
    )

    # Input data
    fig2.add_trace(
        go.Scatter(
            x=time_steps,
            y=input_data[:, var_idx],
            mode='lines+markers',
            name='Input',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )

    # Target data
    fig2.add_trace(
        go.Scatter(
            x=time_steps,
            y=target_data[:, var_idx],
            mode='lines+markers',
            name='Target',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )

    fig2.update_xaxes(title_text="时间步", row=1, col=1)
    fig2.update_xaxes(title_text="时间步", row=2, col=1)
    fig2.update_yaxes(title_text="数值", row=1, col=1)
    fig2.update_yaxes(title_text="数值", row=2, col=1)

    fig2.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig2, width='stretch')

with tab2:
    st.subheader(f"数值列表 - 变量 {var_name}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 Input Data")
        st.write(f"**变量 {var_name} 时间窗口内数值:**")
        input_df = pd.DataFrame({
            '时间步': time_steps,
            '数值': input_data[:, var_idx]
        })
        st.dataframe(input_df, height=600, width='stretch')

    with col2:
        st.markdown("### 🔴 Target Data")
        st.write(f"**变量 {var_name} 时间窗口内数值:**")
        target_df = pd.DataFrame({
            '时间步': time_steps,
            '数值': target_data[:, var_idx]
        })
        st.dataframe(target_df, height=600, width='stretch')

with tab3:
    st.subheader(f"统计信息 - 变量 {var_name}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 Input Data 统计")
        st.write(f"**变量 {var_name} 统计 (时间窗口内):**")
        input_values = input_data[:, var_idx]
        stats_input_var = pd.DataFrame({
            '指标': ['最小值', '最大值', '平均值', '标准差', '中位数'],
            '数值': [
                f"{np.min(input_values):.6f}",
                f"{np.max(input_values):.6f}",
                f"{np.mean(input_values):.6f}",
                f"{np.std(input_values):.6f}",
                f"{np.median(input_values):.6f}"
            ]
        })
        st.dataframe(stats_input_var, width='stretch')

    with col2:
        st.markdown("### 🔴 Target Data 统计")
        st.write(f"**变量 {var_name} 统计 (时间窗口内):**")
        target_values = target_data[:, var_idx]
        stats_target_var = pd.DataFrame({
            '指标': ['最小值', '最大值', '平均值', '标准差', '中位数'],
            '数值': [
                f"{np.min(target_values):.6f}",
                f"{np.max(target_values):.6f}",
                f"{np.mean(target_values):.6f}",
                f"{np.std(target_values):.6f}",
                f"{np.median(target_values):.6f}"
            ]
        })
        st.dataframe(stats_target_var, width='stretch')

    # 差异统计
    st.markdown("---")
    st.markdown("### 📊 Input vs Target 差异")

    diff = target_data[:, var_idx] - input_data[:, var_idx]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("平均差异", f"{np.mean(diff):.6f}")
        st.metric("差异标准差", f"{np.std(diff):.6f}")

    with col2:
        st.metric("最大正差异", f"{np.max(diff):.6f}")
        st.metric("最大负差异", f"{np.min(diff):.6f}")

    with col3:
        st.metric("零值比例", f"{np.mean(diff == 0) * 100:.2f}%")
        st.metric("非零差异", f"{np.sum(diff != 0)}")

# 页脚
st.markdown("---")
st.caption("💡 使用左侧边栏选择样本、时间窗口和变量进行查看")
