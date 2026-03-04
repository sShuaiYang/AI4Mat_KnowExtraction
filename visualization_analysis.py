import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.ticker as mtick
import json
from collections import Counter
import warnings
from scipy.interpolate import make_interp_spline
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# plt.rcParams['axes.unicode_minus'] = False
# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_and_prepare_data(file_path):
    """加载并预处理数据"""
    print("Loading data...")
    df = pd.read_excel(file_path)
    
    # 提取年份
    if 'Publication Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Publication Year'], errors='coerce')
    elif 'year' in df.columns:
        df['Year'] = pd.to_numeric(df['year'], errors='coerce')
    else:
        print("Warning: No year column found")
        df['Year'] = 2020  # 默认值
    
    # 过滤掉无效年份
    df = df[df['Year'].notna()]
    df = df[(df['Year'] >= 2000) & (df['Year'] <= 2026)]
    
    print(f"Loaded {len(df)} papers")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    
    return df


def plot_yearly_growth(df, output_path='fig1_yearly_growth.png'):
    """
    图1: 材料信息提取文献逐年增长趋势（数量+增长率）
    """
    print("\n[Figure 1] Creating yearly growth trend plot...")
    
    yearly_counts = df.groupby('Year').size().reset_index(name='Count')
    yearly_counts['Growth_Rate'] = yearly_counts['Count'].pct_change() * 100
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # 左侧Y轴：论文数量
    color1 = '#2E86AB'
    ax1.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Papers', fontsize=14, fontweight='bold', color=color1)
    
    bars = ax1.bar(yearly_counts['Year'], yearly_counts['Count'], 
                   color=color1, alpha=0.7, edgecolor='black', linewidth=0.8)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12, rotation=45)
    
    # 在柱状图上标注数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 右侧Y轴：增长率
    ax2 = ax1.twinx()
    color2 = '#A23B72'
    ax2.set_ylabel('Growth Rate (%)', fontsize=14, fontweight='bold', color=color2)
    
    # 绘制增长率折线
    line = ax2.plot(yearly_counts['Year'][1:], yearly_counts['Growth_Rate'][1:], 
                    color=color2, marker='o', linewidth=2.5, markersize=8,
                    label='Growth Rate', markeredgecolor='white', markeredgewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 标题和网格
    plt.title('Annual Growth Trend of Materials Data Extraction Publications', 
              fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # 添加统计信息
    total_papers = len(df)
    recent_5years = len(df[df['Year'] >= df['Year'].max() - 4])
    textstr = f'Total Papers: {total_papers}\nRecent 5 Years: {recent_5years} ({recent_5years/total_papers*100:.1f}%)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    pdf_path = output_path.replace('.png', '.pdf').replace('.jpg', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_model_category_pie(df, output_path='fig2_model_category_pie.png'):
    """
    图2: 模型分类占比饼图（突出显示LLM）
    """
    print("\n[Figure 2] Creating model category pie chart (highlighting LLM)...")
    
    # 统计Primary Category
    if 'model_primary_category' not in df.columns:
        print("Warning: No model_primary_category column found")
        return
    
    category_counts = df['model_primary_category'].value_counts()
    
    # 准备数据
    labels = category_counts.index.tolist()
    sizes = category_counts.values.tolist()
    
    # 突出LLM：使用explode
    explode = []
    colors = []
    for label in labels:
        if 'Large Language' in label: #'LLM' in label or
            explode.append(0.15)  # 突出LLM
            colors.append('#FF6B6B')  # 红色突出
        else:
            explode.append(0.02)
            colors.append(None)
    
    # 如果没有指定颜色，使用默认调色板
    if all(c is None for c in colors):
        colors = sns.color_palette("Set2", len(labels))
    else:
        default_colors = sns.color_palette("Set2", len(labels))
        colors = [c if c else default_colors[i] for i, c in enumerate(colors)]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%',
                                       startangle=90, colors=colors, explode=explode,
                                       shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # 美化百分比文本
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # 添加图例
    legend_labels = [f'{label}: {size} ({size/sum(sizes)*100:.1f}%)' 
                     for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Model Categories", 
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10, title_fontsize=12)
    
    plt.title('Distribution of Data Extraction Models\n(LLM Highlighted)', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_yearly_growth_optimized(df, output_path='fig1_yearly_growth_optimized.png'):
    """
    图1优化版: 全框 + 仅横向网格 + 平滑趋势
    """
    print("\n[Figure 1] Creating optimized yearly growth plot (Full Frame)...")

    # 1. 数据准备
    yearly_counts = df.groupby('Year').size().reset_index(name='Count')
    yearly_counts = yearly_counts.sort_values('Year')

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 7))

    # 2. 设置网格 (关键修改：只开Y轴网格，不开X轴)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray', zorder=0)
    ax.set_axisbelow(True)  # 让网格线在柱子下面

    # 3. 绘制柱状图
    bar_color = '#4c72b0'
    bars = ax.bar(yearly_counts['Year'], yearly_counts['Count'],
                  color=bar_color, alpha=0.6, width=0.7,
                  edgecolor='black', linewidth=0.5,  # 柱子加个细边框更清晰
                  label='Number of Papers', zorder=2)

    # 4. 绘制平滑趋势线
    if len(yearly_counts) > 3:
        x = yearly_counts['Year'].values
        y = yearly_counts['Count'].values

        x_smooth = np.linspace(x.min(), x.max(), 300)
        try:
            spl = make_interp_spline(x, y, k=3)
            y_smooth = spl(x_smooth)
            y_smooth = np.maximum(y_smooth, 0)

            line_color = '#c44e52'
            ax.plot(x_smooth, y_smooth, color=line_color, linewidth=3,
                    alpha=0.9, label='Growth Trend', zorder=3)
        except Exception:
            ax.plot(x, y, color='#c44e52', linewidth=2, marker='o', zorder=3)
    else:
        ax.plot(yearly_counts['Year'], yearly_counts['Count'], color='#c44e52', linewidth=2, marker='o', zorder=3)

    # 5. 标注数值
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10,
                    color='black', fontweight='bold', zorder=4)

    # 6. 边框与刻度设置 (关键修改：全框显示)
    # 显式设置四个边框为可见，且颜色为黑色
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.2)  # 稍微加粗一点点，更有质感
        ax.spines[spine].set_color('black')

    ax.set_xlabel('Year', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of Publications', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Annual Growth Trend of Materials Data Extraction',
                 fontsize=16, fontweight='bold', pad=20)

    # 调整X轴刻度
    plt.xticks(yearly_counts['Year'], rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2, color='black')  # 刻度线也匹配边框颜色

    # 添加图例
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='black', fontsize=11)

    # 添加统计信息框
    total_papers = len(df)
    props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=1.0)
    ax.text(0.98, 0.95, f'Total Publications: {total_papers}',
            transform=ax.transAxes, fontsize=11, fontweight='bold', color='black',
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf').replace('.jpg', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_model_category_optimized(df, output_path='fig2_model_category_donut.png'):
    """
    图2优化版: 甜甜圈图 + 自动合并小类别 + 智能图例
    """
    print("\n[Figure 2] Creating optimized model category donut chart...")

    if 'model_primary_category' not in df.columns:
        print("Warning: No model_primary_category column found")
        return

    # 1. 数据处理：合并小类别 (The "Clean Up" Logic)
    counts = df['model_primary_category'].value_counts()
    total = counts.sum()

    # 设定阈值：比如占比小于 3% 的归为 "Others"
    threshold_pct = 0.03

    # 找出主要类别
    main_categories = counts[counts / total >= threshold_pct].copy()

    # 找出小类别
    small_categories = counts[counts / total < threshold_pct]

    # 强制逻辑：如果 LLM 被归为了小类别（虽然不太可能），我们要把它捞回来
    # 假设 LLM 的关键字是 'LLM' 或 'Language'
    llm_mask = small_categories.index.str.contains('Large Language', case=False, regex=True)
    if llm_mask.any():
        llm_series = small_categories[llm_mask]
        main_categories = pd.concat([main_categories, llm_series])
        small_categories = small_categories[~llm_mask]  # 从小类别中移除 LLM

    # 计算 Others 的总和
    other_count = small_categories.sum()

    # 重新构建绘图数据
    plot_data = main_categories.sort_values(ascending=False)  # 排序
    if other_count > 0:
        plot_data['Others'] = other_count

    labels = plot_data.index.tolist()
    sizes = plot_data.values.tolist()

    # 2. 颜色设置 (自动突出 LLM)
    # 使用 Seaborn 的 Set3 或 Pastel 调色板，看起来不刺眼
    base_colors = sns.color_palette("Pastel1", len(labels))
    colors = list(base_colors)

    explode = [0] * len(labels)

    for i, label in enumerate(labels):
        if 'LLM' in label or 'Large Language' in label:
            colors[i] = '#ff6b6b'  # 鲜艳的红/珊瑚色
            explode[i] = 0.05  # 稍微炸开一点点
        elif label == 'Others':
            colors[i] = '#e0e0e0'  # 灰色用于 Others

    # 3. 绘制甜甜圈图
    fig, ax = plt.subplots(figsize=(10, 6))

    # wedgeprops width=0.4 制造甜甜圈效果
    wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='%1.1f%%',
                                      startangle=140, colors=colors, explode=explode,
                                      pctdistance=0.80,  # 百分比距离圆心的距离
                                      wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))

    # 4. 美化文本
    plt.setp(autotexts, size=9, weight="bold", color="#333333")

    # 对于太小的扇区，隐藏饼图上的百分比，依靠图例
    for i, autotext in enumerate(autotexts):
        if sizes[i] / sum(sizes) < 0.04:  # 小于4%的不在图上显示字，太挤了
            autotext.set_visible(False)
        # 如果是 LLM，强制显示并变白
        if explode[i] > 0:
            autotext.set_visible(True)
            autotext.set_color('white')

    # 5. 中间添加文字
    ax.text(0, 0, 'Model\nTypes', ha='center', va='center', fontsize=14, fontweight='bold', color='#555555')

    # 6. 右侧详细图例
    # 计算百分比用于图例
    total_plot = sum(sizes)
    legend_labels = [f'{label} ({size / total_plot:.1%})' for label, size in zip(labels, sizes)]

    ax.legend(wedges, legend_labels,
              title="Category Distribution",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),  # 将图例移到图表外面
              frameon=False,
              fontsize=10)

    ax.set_title('Distribution of Extraction Models\n(LLM Highlighted)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf').replace('.jpg', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_llm_trend_over_years(df, output_path='fig3_llm_trend.png'):
    """
    图3: LLM在各年份的占比趋势（堆叠面积图）
    """
    print("\n[Figure 3] Creating LLM trend over years...")
    
    if 'model_primary_category' not in df.columns:
        print("Warning: No model_primary_category column found")
        return
    
    # 创建LLM标记
    df['is_LLM'] = df['model_primary_category'].apply(
        lambda x: 'LLM' if pd.notna(x) and ('LLM' in str(x) or 'Large Language' in str(x)) else 'Non-LLM'
    )
    
    # 按年份和LLM分类统计
    yearly_llm = df.groupby(['Year', 'is_LLM']).size().unstack(fill_value=0)
    
    # 如果没有LLM列，添加它
    if 'LLM' not in yearly_llm.columns:
        yearly_llm['LLM'] = 0
    if 'Non-LLM' not in yearly_llm.columns:
        yearly_llm['Non-LLM'] = 0
    
    # 计算百分比
    yearly_llm_pct = yearly_llm.div(yearly_llm.sum(axis=1), axis=0) * 100
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), 
                                     gridspec_kw={'height_ratios': [2, 1]})
    
    # 子图1: 堆叠面积图
    colors_area = ['#FF6B6B', '#A8DADC']
    ax1.fill_between(yearly_llm.index, 0, yearly_llm['LLM'], 
                     color=colors_area[0], alpha=0.7, label='LLM')
    ax1.fill_between(yearly_llm.index, yearly_llm['LLM'], 
                     yearly_llm['LLM'] + yearly_llm['Non-LLM'],
                     color=colors_area[1], alpha=0.7, label='Non-LLM')
    
    ax1.set_ylabel('Number of Papers', fontsize=14, fontweight='bold')
    ax1.set_title('Evolution of LLM vs Non-LLM Models in Materials Data Extraction', 
                  fontsize=16, fontweight='bold', pad=15)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # 在堆叠图上标注具体数值
    for year in yearly_llm.index:
        llm_count = yearly_llm.loc[year, 'LLM']
        total_count = yearly_llm.loc[year].sum()
        if llm_count > 0:
            ax1.text(year, llm_count/2, f'{int(llm_count)}', 
                    ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # 子图2: LLM占比趋势线
    ax2.plot(yearly_llm_pct.index, yearly_llm_pct['LLM'], 
            color='#FF6B6B', marker='o', linewidth=3, markersize=10,
            markeredgecolor='white', markeredgewidth=2, label='LLM Percentage')
    ax2.fill_between(yearly_llm_pct.index, 0, yearly_llm_pct['LLM'], 
                     color='#FF6B6B', alpha=0.2)
    
    # 标注百分比数值
    for year, pct in yearly_llm_pct['LLM'].items():
        if pct > 0:
            ax2.text(year, pct + 2, f'{pct:.1f}%', 
                    ha='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax2.set_ylabel('LLM Percentage (%)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% Threshold')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    pdf_path = output_path.replace('.png', '.pdf').replace('.jpg', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()


def plot_llm_trend_optimized(df, output_path='fig3_llm_trend_optimized.png'):
    """
    图3优化版:
    上图：堆叠柱状图 (数量) - 视觉清晰
    下图：折线面积图 (占比) - 趋势直观
    特点：共用X轴，全框风格，配色统一
    """
    print("\n[Figure 3] Creating optimized LLM trend plot...")

    if 'model_primary_category' not in df.columns:
        print("Warning: No model_primary_category column found")
        return

    # 1. 数据准备
    df = df.copy()  # 防止修改原始数据
    df['is_LLM'] = df['model_primary_category'].apply(
        lambda x: 'LLM' if pd.notna(x) and ('LLM' in str(x) or 'Large Language' in str(x)) else 'Non-LLM'
    )

    # 统计数量
    yearly_counts = df.groupby(['Year', 'is_LLM']).size().unstack(fill_value=0)

    # 补全列防止报错
    for col in ['Non-LLM', 'LLM']:
        if col not in yearly_counts.columns:
            yearly_counts[col] = 0

    # 计算百分比
    yearly_total = yearly_counts.sum(axis=1)
    llm_pct = (yearly_counts['LLM'] / yearly_total) * 100

    # 2. 创建画布 (共用X轴)
    # height_ratios=[2, 1.2] 让上图稍微大一点，下图扁一点
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1.2], 'hspace': 0.05})

    # 定义统一颜色
    color_llm = '#c44e52'  # 砖红色 (与之前趋势线一致)
    color_non_llm = '#4c72b0'  # 蓝色 (与之前柱状图一致)

    # --- 上图：堆叠柱状图 (数量) ---
    # 使用柱状图比面积图在离散年份上更严谨
    ax1.bar(yearly_counts.index, yearly_counts['Non-LLM'],
            label='Traditional Models', color=color_non_llm, alpha=0.7,
            edgecolor='black', linewidth=0.5, width=0.6)

    ax1.bar(yearly_counts.index, yearly_counts['LLM'], bottom=yearly_counts['Non-LLM'],
            label='LLM-based Models', color=color_llm, alpha=0.8,
            edgecolor='black', linewidth=0.5, width=0.6)

    # 在柱子里标注数值 (仅当数值足够大时)
    for i, year in enumerate(yearly_counts.index):
        # 标 Non-LLM
        v_non = yearly_counts.loc[year, 'Non-LLM']
        if v_non > 0:
            ax1.text(year, v_non / 2, int(v_non), ha='center', va='center',
                     color='white', fontsize=9, fontweight='bold')

        # 标 LLM
        v_llm = yearly_counts.loc[year, 'LLM']
        if v_llm > 0:
            # 位置是底部高度 + 自身高度的一半
            pos_y = v_non + v_llm / 2
            ax1.text(year, pos_y, int(v_llm), ha='center', va='center',
                     color='white', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Number of Papers', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    ax1.set_title('Evolution of Extraction Models: Traditional vs LLM',
                  fontsize=16, fontweight='bold', pad=15)

    # --- 下图：占比趋势图 (百分比) ---
    # 绘制占比线
    ax2.plot(llm_pct.index, llm_pct, color=color_llm, marker='o',
             linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=1.5,
             label='LLM Adoption Rate')

    # 填充下方区域
    ax2.fill_between(llm_pct.index, 0, llm_pct, color=color_llm, alpha=0.15)

    # 标注百分比 (智能避让，太低的不标)
    for year, pct in llm_pct.items():
        if pct > 2:  # 只有大于2%才标注
            ax2.text(year, pct + 3, f'{pct:.0f}%', ha='center', va='bottom',
                     fontsize=9, color=color_llm, fontweight='bold')

    # 添加 50% 参考线
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(yearly_counts.index.min(), 52, 'Dominance Threshold (50%)',
             color='gray', fontsize=8, style='italic')

    ax2.set_ylabel('LLM Share (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)

    # 格式化Y轴为百分比
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

    # --- 通用美化 (全框 + 网格) ---
    for ax in [ax1, ax2]:
        # 只开横向网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray', zorder=0)
        ax.set_axisbelow(True)

        # 全框边框设置
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(1.2)
            ax.spines[spine].set_color('black')

        ax.tick_params(axis='both', colors='black', width=1.2)

    # 调整X轴刻度 (因为 sharex，只在 ax2 设置即可，但需要确保刻度对其)
    plt.xticks(yearly_counts.index, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = output_path.replace('.png', '.pdf').replace('.jpg', '.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_material_category_distribution(df, output_path='fig4_material_distribution.png'):
    """
    图4: 材料类别分布（横向条形图）
    """
    print("\n[Figure 4] Creating material category distribution...")
    
    if 'primary_material_category' not in df.columns:
        print("Warning: No primary_material_category column found")
        return
    
    # 统计材料类别
    material_counts = df['primary_material_category'].value_counts().head(12)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建渐变色
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(material_counts)))
    
    # 绘制横向条形图
    bars = ax.barh(range(len(material_counts)), material_counts.values, 
                   color=colors, edgecolor='black', linewidth=0.8)
    
    ax.set_yticks(range(len(material_counts)))
    ax.set_yticklabels(material_counts.index, fontsize=11)
    ax.set_xlabel('Number of Papers', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Primary Material Categories', 
                 fontsize=15, fontweight='bold', pad=15)
    
    # 在条形上标注数值
    for i, (bar, count) in enumerate(zip(bars, material_counts.values)):
        width = bar.get_width()
        ax.text(width + max(material_counts.values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{int(count)}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()  # 最大值在上
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_model_material_heatmap(df, output_path='fig5_model_material_heatmap.png'):
    """
    图5: 模型类别与材料类别的关联热图
    """
    print("\n[Figure 5] Creating model-material correlation heatmap...")
    
    if 'model_primary_category' not in df.columns or 'primary_material_category' not in df.columns:
        print("Warning: Required columns not found")
        return
    
    # 创建交叉表
    crosstab = pd.crosstab(df['model_primary_category'], 
                          df['primary_material_category'])
    
    # 只保留top的类别以便可视化
    top_models = df['model_primary_category'].value_counts().head(8).index
    top_materials = df['primary_material_category'].value_counts().head(10).index
    
    crosstab_filtered = crosstab.loc[top_models, top_materials]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制热图
    sns.heatmap(crosstab_filtered, annot=True, fmt='d', cmap='YlOrRd',
                linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Number of Papers'},
                ax=ax, annot_kws={'fontsize': 9, 'fontweight': 'bold'})
    
    ax.set_xlabel('Material Category', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model Category', fontsize=13, fontweight='bold')
    ax.set_title('Correlation Between Model Types and Material Categories', 
                 fontsize=15, fontweight='bold', pad=15)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_confidence_distribution(df, output_path='fig6_confidence.png'):
    """
    图6: 模型和材料分类的置信度分布
    """
    print("\n[Figure 6] Creating confidence distribution plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 模型分类置信度
    if 'model_confidence' in df.columns:
        model_conf = df['model_confidence'].value_counts()
        colors1 = ['#2ecc71', '#f39c12', '#e74c3c']
        wedges1, texts1, autotexts1 = ax1.pie(model_conf.values, labels=model_conf.index,
                                               autopct='%1.1f%%', startangle=90,
                                               colors=colors1, textprops={'fontsize': 11})
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax1.set_title('Model Classification\nConfidence', fontsize=13, fontweight='bold')
    
    # 材料分类置信度
    if 'material_confidence' in df.columns:
        material_conf = df['material_confidence'].value_counts()
        colors2 = ['#2ecc71', '#f39c12', '#e74c3c']
        wedges2, texts2, autotexts2 = ax2.pie(material_conf.values, labels=material_conf.index,
                                               autopct='%1.1f%%', startangle=90,
                                               colors=colors2, textprops={'fontsize': 11})
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax2.set_title('Material Classification\nConfidence', fontsize=13, fontweight='bold')
    
    plt.suptitle('Classification Confidence Distribution', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_top_journals(df, output_path='fig7_top_journals.png'):
    """
    图7: 发表期刊Top 15
    """
    print("\n[Figure 7] Creating top journals plot...")
    
    journal_col = None
    for col in ['Source Title', 'Journal', 'Publication Name']:
        if col in df.columns:
            journal_col = col
            break
    
    if journal_col is None:
        print("Warning: No journal column found")
        return
    
    top_journals = df[journal_col].value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(top_journals)))
    bars = ax.barh(range(len(top_journals)), top_journals.values,
                   color=colors, edgecolor='black', linewidth=0.8)
    
    ax.set_yticks(range(len(top_journals)))
    ax.set_yticklabels(top_journals.index, fontsize=10)
    ax.set_xlabel('Number of Papers', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Journals Publishing Materials Data Extraction Research', 
                 fontsize=14, fontweight='bold', pad=15)
    
    for i, (bar, count) in enumerate(zip(bars, top_journals.values)):
        width = bar.get_width()
        ax.text(width + max(top_journals.values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{int(count)}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_multi_category_analysis(df, output_path='fig8_multi_category.png'):
    """
    图8: 多类别材料分析（一篇文献涉及多种材料的情况）
    """
    print("\n[Figure 8] Creating multi-category analysis...")
    
    if 'material_categories' not in df.columns:
        print("Warning: No material_categories column found")
        return
    
    # 统计每篇文献涉及的材料类别数量
    category_counts = []
    for cats_str in df['material_categories'].dropna():
        try:
            cats = json.loads(cats_str)
            category_counts.append(len(cats))
        except:
            pass
    
    if not category_counts:
        print("Warning: No valid category data")
        return
    
    count_dist = pd.Series(category_counts).value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(count_dist)))
    bars = ax.bar(count_dist.index, count_dist.values, color=colors,
                  edgecolor='black', linewidth=1.2, width=0.6)
    
    # 标注数值和百分比
    total = sum(count_dist.values)
    for bar, count in zip(bars, count_dist.values):
        height = bar.get_height()
        percentage = count / total * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of Material Categories per Paper', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Papers', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Multiple Material Categories in Single Papers', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(count_dist.index)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_summary_statistics(df, output_path='summary_statistics.txt'):
    """
    生成汇总统计报告
    """
    print("\n[Summary] Generating statistics report...")
    
    report = []
    report.append("="*80)
    report.append("MATERIALS DATA EXTRACTION LITERATURE ANALYSIS SUMMARY")
    report.append("="*80)
    report.append("")
    
    # 基本统计
    report.append("1. BASIC STATISTICS")
    report.append("-" * 80)
    report.append(f"Total Papers: {len(df)}")
    report.append(f"Year Range: {int(df['Year'].min())} - {int(df['Year'].max())}")
    report.append(f"Average Papers per Year: {len(df) / (df['Year'].max() - df['Year'].min() + 1):.1f}")
    report.append("")
    
    # 年份统计
    report.append("2. RECENT GROWTH (Last 5 Years)")
    report.append("-" * 80)
    recent_years = df[df['Year'] >= df['Year'].max() - 4]
    report.append(f"Papers (Last 5 Years): {len(recent_years)} ({len(recent_years)/len(df)*100:.1f}%)")
    for year in sorted(recent_years['Year'].unique(), reverse=True):
        year_count = len(recent_years[recent_years['Year'] == year])
        report.append(f"  {int(year)}: {year_count} papers")
    report.append("")
    
    # 模型分类统计
    if 'model_primary_category' in df.columns:
        report.append("3. MODEL CATEGORY DISTRIBUTION")
        report.append("-" * 80)
        model_counts = df['model_primary_category'].value_counts()
        for cat, count in model_counts.head(10).items():
            pct = count / len(df) * 100
            report.append(f"  {cat:<50} {count:>4} ({pct:>5.1f}%)")
        
        # LLM统计
        llm_count = sum([count for cat, count in model_counts.items() 
                        if 'LLM' in str(cat) or 'Large Language' in str(cat)])
        report.append(f"\n  Total LLM Papers: {llm_count} ({llm_count/len(df)*100:.1f}%)")
        report.append("")
    
    # 材料分类统计
    if 'primary_material_category' in df.columns:
        report.append("4. MATERIAL CATEGORY DISTRIBUTION")
        report.append("-" * 80)
        material_counts = df['primary_material_category'].value_counts()
        for cat, count in material_counts.head(10).items():
            pct = count / len(df) * 100
            report.append(f"  {cat:<50} {count:>4} ({pct:>5.1f}%)")
        report.append("")
    
    # 置信度统计
    if 'model_confidence' in df.columns:
        report.append("5. CLASSIFICATION CONFIDENCE")
        report.append("-" * 80)
        report.append("Model Classification:")
        model_conf = df['model_confidence'].value_counts()
        for conf, count in model_conf.items():
            report.append(f"  {conf}: {count} ({count/len(df)*100:.1f}%)")
        
        if 'material_confidence' in df.columns:
            report.append("\nMaterial Classification:")
            material_conf = df['material_confidence'].value_counts()
            for conf, count in material_conf.items():
                report.append(f"  {conf}: {count} ({count/len(df)*100:.1f}%)")
        report.append("")
    
    report.append("="*80)
    
    # 打印到控制台
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # 保存到文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n✓ Saved: {output_path}")


def create_all_visualizations(file_path):
    """
    创建所有可视化图表
    """
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE VISUALIZATION ANALYSIS")
    print("="*80)
    
    # 加载数据
    df = load_and_prepare_data(file_path)
    
    # 创建输出目录
    import os
    output_dir = './visualization_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成所有图表
    plot_yearly_growth(df, f'{output_dir}/fig1_yearly_growth.png')
    plot_yearly_growth_optimized(df, f'{output_dir}/fig2_yearly_growth_optimized.png')
    plot_model_category_pie(df, f'{output_dir}/fig2_model_category_pie.png')
    plot_model_category_optimized(df, f'{output_dir}/fig3_model_category_optimized.png')
    plot_llm_trend_over_years(df, f'{output_dir}/fig3_llm_trend.png')
    plot_llm_trend_optimized(df, f'{output_dir}/fig4_llm_trend_optimized.png')
    plot_material_category_distribution(df, f'{output_dir}/fig4_material_distribution.png')
    plot_model_material_heatmap(df, f'{output_dir}/fig5_model_material_heatmap.png')
    plot_confidence_distribution(df, f'{output_dir}/fig6_confidence.png')
    plot_top_journals(df, f'{output_dir}/fig7_top_journals.png')
    plot_multi_category_analysis(df, f'{output_dir}/fig8_multi_category.png')
    
    # 生成统计报告
    generate_summary_statistics(df, f'{output_dir}/summary_statistics.txt')
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETED!")
    print(f"✓ Results saved to: {output_dir}/")
    print("="*80)
    
    return df


if __name__ == "__main__":
    # 运行分析
    file_path = r"G:\2026-01-26 材料信息提取\combined_classified_with_materials_true.xlsx"
    df = create_all_visualizations(file_path)
