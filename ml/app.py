from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import matplotlib
matplotlib.use('Agg')
import subprocess, time
import pandas as pd
from pymatgen.core.composition import Composition
import re,json,logging
import requests
import os
#from pymatgen.ext.matproj import MPRester
from mp_api.client import MPRester
import re
app = Flask(__name__)


# 配置 CORS（允许前端跨域访问）
CORS(app, resources=r'/*')



# 设置日志记录级别为 INFO
app.logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')




API_KEY = ""


def fetch_material_properties(formula):
    try:
        with MPRester(API_KEY) as mpr:
            results = mpr.summary.search(formula=formula)
            if not results:
                print(f"未找到 {formula} 的材料数据。")
                return None

            # 默认取第一个结果
            material = results[0]
            return {
                "band_gap": material.band_gap,
                "density": material.density,
                "cbm": getattr(material, "cbm", None),
                "vbm": getattr(material, "vbm", None),
                "formation_energy_per_atom": material.formation_energy_per_atom,
                "energy_per_atom": material.energy_per_atom,
                "efermi": getattr(material, "efermi", None),
            }
    except Exception as e:
        print("Error fetching from Materials Project:", e)
        return None


import time
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import cloudinary
import cloudinary.uploader

from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


cloudinary.config(
    cloud_name="",
    api_key="",
    api_secret="",
    secure=True
)

def upload_to_cloudinary(image_path, public_id):
    result = cloudinary.uploader.upload(image_path, public_id=public_id, overwrite=True)
    return result["secure_url"]

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.tsne_result = None
        self.center_point = None
        self.tsne_model = None
        self.pca_model = None
        self.scaler = None

    def load_data(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            data = pd.read_csv(file_path)
            columns_lower = [col.lower() for col in data.columns]
            if 'mae' not in columns_lower or 'mse' not in columns_lower:
                raise ValueError("数据文件必须包含MAE和MSE列（不区分大小写）")
            mae_col = data.columns[columns_lower.index('mae')]
            mse_col = data.columns[columns_lower.index('mse')]
            feature_cols = [col for col in data.columns if col.lower() not in ['mae', 'mse'] and col != data.columns[-3]]
            if len(feature_cols) < 1:
                raise ValueError("数据必须至少包含一个特征列")
            missing_info = []
            for col in data.columns:
                missing_count = data[col].isnull().sum()
                if missing_count > 0:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        mean_val = data[col].mean()
                        data[col] = data[col].fillna(mean_val)
                        missing_info.append(f"{col}: {missing_count}个缺失值已用平均值{mean_val:.4f}填充")
                    else:
                        mode_val = data[col].mode()[0]
                        data[col].fillna(mode_val, inplace=True)
                        missing_info.append(f"{col}: {missing_count}个缺失值已用众数'{mode_val}'填充")
            if mae_col != 'MAE' or mse_col != 'MSE':
                data = data.rename(columns={mae_col: 'MAE', mse_col: 'MSE'})
            self.data = data
            info_message = f"已加载数据，共{len(data)}行\n特征数量：{len(feature_cols)}"
            if missing_info:
                info_message += "\n\n处理的缺失值："
                info_message += "\n".join(missing_info)
            print(info_message)
        except Exception as e:
            raise ValueError(f"加载数据失败: {str(e)}")

    def run_tsne(self, perplexity=19, n_components=2, random_state=42):
        if self.data is None:
            raise ValueError("请先加载数据")
        if perplexity >= len(self.data):
            raise ValueError(f"Perplexity值({perplexity})不能大于或等于数据样本数({len(self.data)})。")
        feature_cols = [col for col in self.data.columns if col not in ['MAE', 'MSE']]
        X = self.data[feature_cols]
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        self.tsne_result = tsne.fit_transform(X)
        self.tsne_model = tsne

    def fill_missing_with_train_values(self, test_data):
        filled_data = test_data.copy()
        for col in self.data.columns:
            if col not in filled_data.columns:
                continue  # 测试集没有该列，跳过
            if filled_data[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    mean_val = self.data[col].mean()
                    filled_data[col] = filled_data[col].fillna(mean_val)
                else:
                    mode_val = self.data[col].mode()[0]
                    filled_data[col] = filled_data[col].fillna(mode_val)
        return filled_data

    def plot_results(self, filename="t-sne_distribution.png"):
        if self.tsne_result is None:
            raise ValueError("请先运行t-SNE或PCA分析")
        try:
            plt.rcParams.update({
                'font.family': 'Arial',
                'font.size': 20,
                'axes.titlesize': 20,
                'axes.labelsize': 20,
                'xtick.labelsize': 20,
                'ytick.labelsize': 20
            })
            fig, ax = plt.subplots(figsize=(10, 8))
            mae_values = self.data['MAE']
            print(f"MAE 统计: min={mae_values.min():.4f}, max={mae_values.max():.4f}, "
                  f"mean={mae_values.mean():.4f}, median={mae_values.median():.4f}")
            base_color = '#C0392B'
            colors = ['#1E8BC3', base_color]
            cmap_name = 'custom_cmap'
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
            plt.register_cmap(cmap=cm)
            mae_transformed = np.log1p(mae_values)
            sc = ax.scatter(self.tsne_result[:, 0],
                            self.tsne_result[:, 1],
                            alpha=0.5,
                            c=mae_transformed,
                            cmap=cmap_name,
                            picker=True,
                            pickradius=5)
            center_x = np.mean(self.tsne_result[:, 0])
            center_y = np.mean(self.tsne_result[:, 1])
            self.center_point = (center_x, center_y)
            ax.set_xlabel("Dimension 1", fontsize=24, labelpad=10, fontname='Arial')
            ax.set_ylabel("Dimension 2", fontsize=24, labelpad=10, fontname='Arial')
            cbar = plt.colorbar(sc)
            cbar.set_label('Log(MAE + 1)', rotation=270, labelpad=35, fontsize=20, fontname='Arial')
            cbar.ax.tick_params(labelsize=15)
            ax.tick_params(axis='both', which='major', labelsize=24, pad=8)
            fig.suptitle("Distribution Analysis (MAE-based)", y=0.95, fontsize=24, fontname='Arial')
            fig.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图片已成功导出为 {filename}")
        except Exception as e:
            raise ValueError(f"绘图时发生错误: {str(e)}")

    def load_test_data(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            test_data = pd.read_csv(file_path)
            train_feature_cols = [col for col in self.data.columns if col not in ['MAE', 'MSE']]
            missing_cols = [col for col in train_feature_cols if col not in test_data.columns]
            if missing_cols:
                raise ValueError(f"测试集缺少以下特征列：{', '.join(missing_cols)}")
            missing_info = []
            for col in test_data.columns:
                missing_count = test_data[col].isnull().sum()
                if missing_count > 0:
                    if pd.api.types.is_numeric_dtype(test_data[col]):
                        mean_val = test_data[col].mean()
                        test_data[col].fillna(mean_val, inplace=True)
                        missing_info.append(f"{col}: {missing_count}个缺失值已用平均值{mean_val:.4f}填充")
                    else:
                        mode_val = test_data[col].mode()[0]
                        test_data[col].fillna(mode_val, inplace=True)
                        missing_info.append(f"{col}: {missing_count}个缺失值已用众数'{mode_val}'填充")
            info_message = f"已加载测试数据，共{len(test_data)}行"
            if missing_info:
                info_message += "\n\n处理的缺失值："
                info_message += "\n".join(missing_info)
            print(info_message)
            return test_data
        except Exception as e:
            raise ValueError(f"加载测试数据失败: {str(e)}")

    def run_test_analysis(self, test_data, perplexity=19, random_state=42, filename="test_data_analysis.png",
                          public_id="test_data_plot"):
        if (self.data is None or self.tsne_result is None):
            raise ValueError("请先完成训练集分析")
        if test_data is None:
            raise ValueError("请先加载测试数据")
        try:
            # 1. 填充测试集缺失值，使用训练集统计值
            test_data = self.fill_missing_with_train_values(test_data)

            # 2. 获取特征列（排除 MAE/MSE）
            feature_cols = [col for col in self.data.columns if
                            col not in ['MAE', 'MSE', 'Catalysts_Component_Encoded']]

            feature_cols = [col for col in feature_cols if col in test_data.columns]  # 双保险

            # 3. 数据准备
            X_test = test_data[feature_cols].values
            X_train = self.data[feature_cols].values

            # 4. TSNE 降维
            if perplexity >= len(test_data):
                X_combined = np.vstack((X_train, X_test))
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
                combined_result = tsne.fit_transform(X_combined)
                test_result = combined_result[len(self.data):]
            else:
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
                test_result = tsne.fit_transform(X_test)

            url = self.plot_test_results_and_upload(test_result, filename, public_id)
            return url

        except Exception as e:
            raise ValueError(f"分析过程中发生错误: {str(e)}")

    def plot_test_results_and_upload(self, test_result, filename, public_id):
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 6))
            train_color = '#1E8BC3'
            test_color = '#C0392B'
            ax.scatter(self.tsne_result[:, 0], self.tsne_result[:, 1], c=train_color, alpha=0.8, s=90,
                       marker='o', edgecolor='white', linewidth=0.5, label='Training Set')
            ax.scatter(test_result[:, 0], test_result[:, 1], color=test_color, alpha=0.6, s=100,
                       marker='^', edgecolor='white', linewidth=0.5, label='Prediction Set')
            for i, point in enumerate(test_result):
                ax.text(point[0], point[1], str(i + 1), fontsize=16, ha='center', va='bottom', color='black')
            ax.set_xlabel("Dimension 1", fontsize=18, fontname='Arial')
            ax.set_ylabel("Dimension 2", fontsize=18, fontname='Arial')
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            legend = ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), framealpha=0.8,
                               prop={'family': 'Arial', 'size': 16})
            legend.get_title().set_fontsize(14)
            legend.get_title().set_fontname('Arial')
            fig.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图片已保存为 {filename}")

            # 上传到 Cloudinary
            url = upload_to_cloudinary(filename, public_id)
            print("图片Cloudinary地址：", url)
            return url
        except Exception as e:
            raise ValueError(f"更新图表或上传图床失败: {str(e)}")


    def plot_test_results(self, test_result, filename="test_data_analysis.png"):
        if self.tsne_result is None or self.center_point is None:
            raise ValueError("请先完成训练集分析")
        try:
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 6))
            train_color = '#1E8BC3'
            test_color = '#C0392B'
            ax.scatter(self.tsne_result[:, 0], self.tsne_result[:, 1], c=train_color, alpha=0.8, s=90,
                       marker='o', edgecolor='white', linewidth=0.5, label='Training Set')
            ax.scatter(test_result[:, 0], test_result[:, 1], color=test_color, alpha=0.6, s=100,
                       marker='^', edgecolor='white', linewidth=0.5, label='Prediction Set')
            for i, point in enumerate(test_result):
                ax.text(point[0], point[1], str(i + 1), fontsize=16, ha='center', va='bottom', color='black')
            ax.set_xlabel("Dimension 1", fontsize=18, fontname='Arial')
            ax.set_ylabel("Dimension 2", fontsize=18, fontname='Arial')
            ax.tick_params(axis='x', labelsize=18)
            ax.tick_params(axis='y', labelsize=18)
            legend = ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), framealpha=0.8,
                               prop={'family': 'Arial', 'size': 16})
            legend.get_title().set_fontsize(14)
            legend.get_title().set_fontname('Arial')
            fig.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图片已成功导出为 {filename}")
        except Exception as e:
            raise ValueError(f"更新图表失败: {str(e)}")

# 全局变量用于缓存
_analyzer = None

def process_train(train_file, perplexity=19):
    global _analyzer
    _analyzer = DataAnalyzer()
    _analyzer.load_data(train_file)
    _analyzer.run_tsne(perplexity=perplexity)
    print("训练集已处理并缓存。")

def process_test(test_file, perplexity=19, filename="test_data_visualization.png", public_id="my_test_data_plot"):
    global _analyzer
    if _analyzer is None or _analyzer.data is None or _analyzer.tsne_result is None:
        raise ValueError("请先调用 process_train 处理训练集。")
    test_data = _analyzer.load_test_data(test_file)
    url = _analyzer.run_test_analysis(
        test_data,
        perplexity=perplexity,
        filename=filename,
        public_id=public_id
    )
    return url





def map_file1_to_file2_fixed(file1_path):
    # 文件二完整列顺序
    target_columns = [
        "hydrogen desorption catalysts particle size", "Catalysts_Component_Encoded", "catalytic mass fraction(wt%)",
        "ball milling rotating speed(rpm)", "ball milling time(min)", "CBM", "VBM", "Density (g/cm^3)",
        "Energy per Atom (eV)", "Band Gap (eV)", "Formation Energy (eV/atom)", "Fermi Energy (eV)",
        "Equilibrium Reaction Energy per Atom (eV/atom)", "Catalyst_Category_Encoded", "Metal_Count",
        "ball milling mass ratio", "year", "Volume (cm^3)", "Electronic Energy (eV)", "Total Energy (eV)",
        "Ionic Energy (eV)", "Surface Anisotropy (J/m^2)", "Weighted Surface Energy", "Element_Count",
        "Metal_Fraction", "Contains_Carbon", "Contains_Oxygen", "TM_Count", "C", "Fe", "H", "N", "Ni", "O", "Ti",
        "Onset Temperature", "Predicted_Onset Temperature", "Activation Energy", "Predicted_Activation Energy"
    ]

    # 映射表（文件一字段 -> 文件二字段）
    column_mapping = {
        "hydrogen desorption catalysts particle size": "Hydrogen Desorption Catalysts Particle Size(nm)",
        "Catalysts_Component_Encoded": "Catalysts Component",
        "catalytic mass fraction(wt%)": "Catalytic Mass Fraction(wt%)",
        "ball milling rotating speed(rpm)": "Ball Milling Rotating Speed(rpm)",
        "ball milling time(min)": "Ball Milling Time(min)",
        "CBM": "CBM",
        "VBM": "VBM",
        "Density (g/cm^3)": "Density (g/cm^3)",
        "Energy per Atom (eV)": "Energy Per Atom (eV)",
        "Band Gap (eV)": "Band Gap (eV)",
        "Formation Energy (eV/atom)": "Formation Energy (eV/atom)",
        "Fermi Energy (eV)": "Fermi Energy (eV)",
        "Equilibrium Reaction Energy per Atom (eV/atom)": "Equilibrium Reaction Energy Per Atom (eV/atom)",
        "Catalyst_Category_Encoded": "Catalyst_Category_Encoded",
        "Metal_Count": "Metal_Count",
        "ball milling mass ratio": "Ball Milling Mass Ratio",
        "Electronic Energy (eV)": "Electronic Energy (eV)",
        "Total Energy (eV)": "Total Energy (eV)",
        "Ionic Energy (eV)": "Ionic Energy (eV)",
        "Element_Count": "Element_Count",
        "Metal_Fraction": "Metal_Fraction",
        "Contains_Carbon": "Contains_Carbon",
        "Contains_Oxygen": "Contains_Oxygen",
        "TM_Count": "TM_Count",
        "C": "C",
        "Fe": "Fe",
        "H": "H",
        "N": "N",
        "Ni": "Ni",
        "O": "O",
        "Ti": "Ti",
        "Onset Temperature": "Onset/Initial Dehydrogenation Temperature(°c)",
        "Activation Energy": "Activation Energy(ea)(kj/mol)",
        "Predicted_Onset Temperature": "Predicted_Onset Temperature",
        "Predicted_Activation Energy": "Predicted_Activation Energy"
    }

    # 读取文件一
    df1 = pd.read_csv(file1_path)

    # 构建文件二结构的 DataFrame
    df2 = pd.DataFrame()

    for target_col in target_columns:
        source_col = column_mapping.get(target_col)
        if source_col and source_col in df1.columns:
            df2[target_col] = df1[source_col]
        else:
            df2[target_col] = pd.NA  # 填空白列

    # 导出文件
    output_file = "mapped_output.csv"
    df2.to_csv(output_file, index=False)
    return output_file

# 全局变量用于缓存


# def process_train(train_file, perplexity=19):
#     global _analyzer
#     _analyzer = DataAnalyzer()
#     _analyzer.load_data(train_file)
#     _analyzer.run_tsne(perplexity=perplexity)
#     print("训练集已处理并缓存。")
#
# def process_test(test_file, perplexity=19, filename="test_data_visualization.png", public_id="my_test_data_plot"):
#     global _analyzer
#     if _analyzer is None or _analyzer.data is None or _analyzer.tsne_result is None:
#         raise ValueError("请先调用 process_train 处理训练集。")
#     test_data = _analyzer.load_test_data(test_file)
#     url = _analyzer.run_test_analysis(
#         test_data,
#         perplexity=perplexity,
#         filename=filename,
#         public_id=public_id
#     )
#     return url


material_mapping = {
    # 原有映射
    "graphene": "C", "CNTs": "C", "rGO": "C",
    "MXene": "Ti3C2", "HEA": "", "Gr": "C",
    # 新增映射
    "Titanium": "Ti", "Carbon": "C", "Iron Oxide": "FeO",
    "Nickel Oxide": "NiO", "Vanadium": "V", "Palladium": "Pd",
    "Germanium": "Ge", "activated carbon": "C", "Carbon Black": "C",
    "organosilica": "SiO2", "Ov": "O", "Polystyrene": "",
    "3DG": "C",  # 新增3DG到碳的映射
    "La-Ni@3DG": "La+Ni+C"  # 明确成分解析
}
# 有机物映射
organic_compounds = {
    'polystyrene': 'C8H8', 'cyclohexane': 'C6H12',
    'citric acid': 'C6H8O7', 'stearic acid': 'C18H36O2'
}
# 金属元素集合
metal_elements = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                  'Zr', 'Nb', 'Mo', 'Ag', 'Au', 'Pd', 'Pt', 'Al'}
transition_metals = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
}
def replace_material_names(formula):
    """替换材料名称"""
    for material, replacement in material_mapping.items():
        formula = re.sub(r'\b' + re.escape(material) + r'\b', replacement, formula, flags=re.IGNORECASE)
    return formula
def replace_organic_names(formula):
    """替换有机物名称"""
    for name, formula in organic_compounds.items():
        formula = re.sub(r'\b' + name + r'\b', formula, formula, flags=re.IGNORECASE)
    return formula
def clean_formula(formula):
    """化学式清理增强版"""
    if not isinstance(formula, str) or not formula.strip():
        return ""

    # 统一字符格式
    formula = formula.translate(str.maketrans('（）％－', '()%-'))

    # 移除括号内容
    formula = re.sub(r'[(\[].*?[)\]]', '', formula)

    # 替换分隔符
    formula = re.sub(r'[-/@&＋+]', '+', formula)

    # 处理掺杂描述
    formula = re.sub(r'(\w+)-?doped', r'\1+', formula, flags=re.IGNORECASE)

    # 处理比例配方
    formula = re.sub(r'(\d+\.?\d*)([A-Z][a-z]?)', lambda m: f"{m.group(2)}{m.group(1)}", formula)

    # 处理合金配方
    formula = re.sub(r'([A-Z][a-z]?)(\d+\.?\d*)',
                     lambda m: f"+{m.group(1)}{m.group(2)}" if m.start() > 0 else f"{m.group(1)}{m.group(2)}",
                     formula)

    # 应用映射
    formula = replace_material_names(formula)
    formula = replace_organic_names(formula)

    # 清理残留字符
    return re.sub(r'[^A-Za-z0-9+_.]', '', formula).strip('+')
def parse_chemical_composition(component):
    """成分解析增强版"""
    cleaned = ""
    try:
        if pd.isna(component) or str(component).lower() in ['nan', 'none']:
            return {}

        cleaned = clean_formula(str(component))
        if not cleaned:
            return {}

        # 处理变量（如Hx→H）
        cleaned = re.sub(r'([A-Z][a-z]?)x', r'\1', cleaned)

        parts = [p for p in cleaned.split('+') if p]
        total_comp = Composition()

        for part in parts:
            if '_' in part:
                base, mult = part.split('_', 1)
                comp = Composition(base) * float(mult)
            else:
                comp = Composition(part)
            total_comp += comp

        return {str(e): total_comp[e] for e in total_comp}

    except Exception as e:
        print(f"Parse error: {repr(component)} → {cleaned} | {str(e)}")
        return {}
def extract_features_from_composition(composition):
    """修复后的特征提取函数"""
    if not isinstance(composition, dict) or len(composition) == 0:
        return {
            'Element_Count': 0,
            'Metal_Fraction': 0.0,
            'Contains_Carbon': 0,
            'Contains_Oxygen': 0,
            'TM_Count': 0
        }

    total = sum(composition.values())
    return {
        'Element_Count': len(composition),
        'Metal_Fraction': sum(v for k, v in composition.items() if k in metal_elements) / total if total > 0 else 0,
        'Contains_Carbon': 1 if 'C' in composition else 0,
        'Contains_Oxygen': 1 if 'O' in composition else 0,
        'TM_Count': sum(1 for e in composition if e in transition_metals)
    }
def one_hot_encode_composition(compositions):
    """
    对化学成分进行 One-Hot 编码。
    """
    all_elements = set()
    for comp in compositions:
        try:
            all_elements.update(comp.keys())
        except AttributeError:
            continue

    # 确保至少包含常见元素
    base_elements = {'C', 'O', 'H', 'N', 'Ti', 'Ni', 'Fe'}
    all_elements = all_elements.union(base_elements)

    feature_matrix = []
    for comp in compositions:
        try:
            feature_vector = [1 if element in comp else 0
                              for element in sorted(all_elements)]
        except TypeError:
            feature_vector = [0] * len(all_elements)
        feature_matrix.append(feature_vector)

    return pd.DataFrame(feature_matrix, columns=sorted(all_elements))
def data_parsers(file_path):
    # 修复数据读取方式
    try:
        dtype_dict = {'Catalysts Component': str}
        dtype_dict.update({str(i): str for i in range(30000)})  # 假设最多100列

        df = pd.read_csv(
            file_path,
            dtype=dtype_dict,
            engine='python',
            na_values=['', 'none', 'nan']
        ).fillna({'catalysts component': ''})

        # 解析成分
        df['Parsed_Composition'] = df['Catalysts Component'].apply(parse_chemical_composition)

        # 提取特征
        df['Features'] = df['Parsed_Composition'].apply(extract_features_from_composition)

        # 将特征展开为单独的列
        features_df = pd.DataFrame(df['Features'].tolist())
        df = pd.concat([df, features_df], axis=1)

        # One-Hot 编码化学成分
        one_hot_features = one_hot_encode_composition(df['Parsed_Composition'].tolist())
        df = pd.concat([df, one_hot_features], axis=1)
        df['onset/initial dehydrogenation temperature(°c)'] = ''
        df['activation energy(ea)(kj/mol)'] = ''
        df['Electronic Energy (eV)'] = ''
        df['Total Energy (eV)'] = ''
        df['Ionic Energy (eV)'] = ''
        # 保存结果到 CSV 文件
        df.to_csv('processed_output.csv', index=False)
        print("处理完成，结果已保存到 processed_output.csv")
        return True
    except:
        return False
def extract_non_empty_columns(file_path):

    output = map_file1_to_file2_fixed("./prediction_n.csv")
    print("映射并导出的文件名:", output)

    # 测试集可以多次传入
    # test_files = [output]
    # for i, test_file in enumerate(test_files):
    #     url = process_test(
    #         test_file,
    #         perplexity=19,
    #         filename=f"test_vis_{i}.png",
    #         public_id=f"my_test_data_plot_{i}"
    #     )
    #     print(f"{test_file} 可视化图像已上传，访问地址：{url}")
    test_file = output  # 你的测试文件路径
    url = process_test(
        test_file,
        perplexity=19,
        filename="test_vis.png",
        public_id="my_test_data_plot"
    )
    print(f"{test_file} 可视化图像已上传，访问地址：{url}")
    df = pd.read_csv(file_path)
    # 选择第一行数据，并去除空值
    non_empty_data = df.iloc[0].dropna()
    # 转换为 JSON 格式
    result_json = non_empty_data.to_dict()
    # 添加状态信息
    return json.dumps({"status": "success", "data": result_json,"image": url }, indent=4)





# ======= 工具函数 =======
@app.after_request
def func_res(resp):
    res = make_response(resp)
    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return res

@app.route('/parse_json', methods=['POST'])
def parse_json():
    # 获取前端发送的 JSON 数据
    data = request.get_json()
    print("********************")
    print(data)
    print("********************")

    intput = data["Catalysts Component"]
    intput1 = data["Hydrogen Desorption Catalysts Particle Size(nm)"]
    intput2 = data["Catalytic Mass Fraction(wt%)"]
    intput3 = data["Ball Milling Rotating Speed(rpm)"]
    intput4 = data["Ball Milling Time(min)"]
    intput5 = data["CBM"]
    intput6 = data["VBM"]
    intput7 = data["Density (g/cm^3)"]
    intput8 = data["Energy Per Atom (eV)"]
    intput9 = data["Band Gap (eV)"]
    intput10 = data["Formation Energy (eV/atom)"]
    intput11 = data["Fermi Energy (eV)"]
    intput12 = data["Equilibrium Reaction Energy Per Atom (eV/atom)"]
    print(intput)
    print(intput1)
    print(intput2)
    print(intput3)
    print(intput4)
    print(intput5)
    print(intput6)
    print(intput7)
    print(intput8)
    print(intput9)
    print(intput10)
    print(intput11)
    print(intput12)

    # 查询 Materials Project 替换部分字段（仅当用户未提供时）
    # 打印替换后的数据方便调试

    mp_data = fetch_material_properties(intput)
    if mp_data:
        print("\n✅ 使用 Materials Project 替换后的字段：")
        if not intput9:
            data["Band Gap (eV)"] = mp_data.get("band_gap", intput9)
            print(f"Band Gap (eV): {data['Band Gap (eV)']}")
        if not intput7:
            data["Density (g/cm^3)"] = mp_data.get("density", intput7)
            print(f"Density (g/cm^3): {data['Density (g/cm^3)']}")
        if not intput5:
            data["CBM"] = mp_data.get("cbm", intput5)
            print(f"CBM: {data['CBM']}")
        if not intput6:
            data["VBM"] = mp_data.get("vbm", intput6)
            print(f"VBM: {data['VBM']}")
        if not intput10:
            data["Formation Energy (eV/atom)"] = mp_data.get("formation_energy_per_atom", intput10)
            print(f"Formation Energy (eV/atom): {data['Formation Energy (eV/atom)']}")
        if not intput8:
            data["Energy Per Atom (eV)"] = mp_data.get("energy_per_atom", intput8)
            print(f"Energy Per Atom (eV): {data['Energy Per Atom (eV)']}")
        if not intput11:
            data["Fermi Energy (eV)"] = mp_data.get("efermi", intput11)
            print(f"Fermi Energy (eV): {data['Fermi Energy (eV)']}\n")
        print(input)
        print(intput1)
        print(intput2)
        print(intput3)
        print(intput4)
        print(intput5)
        print(intput6)
        print(intput7)
        print(intput8)
        print(intput9)
        print(intput10)
        print(intput11)
        print(intput12)















    elements = {
        'Value': [intput, intput1, intput2, intput3, intput4, intput5, intput6, intput7, intput8, intput9, intput10, intput11,
                  intput12, "", "", "", "", "", "", "", ""]
    }
    print("elements 内容如下：")
    for i, val in enumerate(elements['Value']):
        print(f"Value[{i}] = {val}")









    df = pd.DataFrame(elements)
    print("1")
    df = df.T
    print("2")
    df.columns = ['Catalysts Component', 'Hydrogen Desorption Catalysts Particle Size(nm)',
                  'Catalytic Mass Fraction(wt%)'
        , 'Ball Milling Rotating Speed(rpm)', 'Ball Milling Time(min)', 'CBM', 'VBM', 'Density (g/cm^3)'
        , 'Energy Per Atom (eV)', 'Band Gap (eV)', 'Formation Energy (eV/atom)', 'Fermi Energy (eV)'
        , 'Equilibrium Reaction Energy Per Atom (eV/atom)', 'Catalyst_Category_Encoded', 'Metal_Count',
                  "Ball Milling Mass Ratio", "Onset/Initial Dehydrogenation Temperature(°c)"
        , 'Activation Energy(ea)(kj/mol)', "Electronic Energy (eV)", "Total Energy (eV)", "Ionic Energy (eV)"]
    df.drop(df.columns[0], axis=1)
    print("3")
    df.to_csv('data.csv', index=False)
    print("4")
    if data_parsers('data.csv'):
        print("5")

        order = f'python predict2.py --model_path "MultiOutput_Model" --data_path "data.csv"  --output_path "prediction_n.csv"'
        print("6")
        subprocess.run(order, shell=True)
        print("7")
        # time.sleep(2)
        try:
            df = pd.read_csv(f"/home/hongchang/下载/MODEI_3/models/prediction_n.csv")
            if df is not None and not df.empty:
                file_path = "/home/hongchang/下载/MODEI_3/models/prediction_n.csv"
                json_result = extract_non_empty_columns(file_path)
                return json_result
            else:
                return jsonify({"error": "AAA The model cannot be loaded due to data errors!"}),401
        except FileNotFoundError:
            return jsonify({"error": "The output file was not found. Please check the model prediction process."}), 402
    else:
        return jsonify({"error": "666 The model cannot be loaded due to data errors!"}), 403
    # except:
    #     # 如果 "Catalysts Component" 不在 JSON 数据中，返回错误信息
    #     return jsonify({"error": "Catalysts Component 不在 JSON 数据The model cannot be loaded due to data errors!"}), 400


# # 替换为你的 CSV 文件路径
# CSV_PATH = '/home/hongchang/下载/MODEI_3/Decision Tree_train_set_updated.csv'
#
# @app.route('/filter_catalysts', methods=['POST'])
# def filter_catalysts():
#     try:
#         # 加载 CSV 数据
#         df = pd.read_csv(CSV_PATH)
#
#         # 获取筛选阈值
#         data = request.json
#         onset_temp_threshold = data.get('Onset_Temperature')
#         activation_threshold = data.get('Activation_Energy')
#
#         if onset_temp_threshold is None or  activation_threshold is None:
#             return jsonify({'error': 'Missing filter thresholds'}), 400
#
#
#
#         # filtered_df = df[
#         #     (df['Onset Temperature'] < 500) &
#         #     (df['Activation Energy'] < 5)
#         #     ]
#         # print("Filtered count:", len(filtered_df))
#         # print("列名如下：")
#         # print(df.columns.tolist())  # 检查有没有 Onset Temperature 和 Activation Energy
#         #
#         # print("\n字段数据类型：")
#         # print(df.dtypes[['Onset Temperature', 'Activation Energy']])
#         #
#         # print("\n字段统计数据：")
#         # print(df[['Onset Temperature', 'Activation Energy']].describe())
#
#
#
#         # 执行筛选
#         filtered_df = df[
#             (df['Onset Temperature'] < onset_temp_threshold) &
#             (df['Activation Energy'] < activation_threshold)
#         ]
#
#         if filtered_df.empty:
#             return jsonify({'message': 'No matching records found'}), 404
#
#         # 按 Average_MAE 升序排列，选出 MAE 最小的三条
#         result_df = filtered_df.sort_values(by='Average_MAE', ascending=True).head(3)
#
#         # 返回所有字段数据
#         result = result_df.to_dict(orient='records')
#         return jsonify(result), 200
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
# 替换为你的 CSV 文件路径
# 替换为你的 CSV 文件路径
# 替换为你的 CSV 文件路径
OUTPUT_JD_CSV_PATH = '/home/hongchang/下载/MODEI_3/output_jd_merged.csv'
DECISION_TREE_CSV_PATH = '/home/hongchang/下载/MODEI_3/Decision Tree_train_set_updated.csv'

import pandas as pd
import numpy as np
from flask import jsonify, request


@app.route('/filter_catalysts', methods=['POST'])
def filter_catalysts():
    try:
        # 加载两个CSV文件，使用合适的编码
        output_df = pd.read_csv(OUTPUT_JD_CSV_PATH, encoding='latin1')  # 尝试latin1编码
        decision_tree_df = pd.read_csv(DECISION_TREE_CSV_PATH, encoding='latin1')  # 尝试latin1编码

        # 获取筛选阈值
        data = request.json
        onset_temp_threshold = data.get('Onset_Temperature')
        activation_threshold = data.get('Activation_Energy')

        if onset_temp_threshold is None or activation_threshold is None:
            return jsonify({'error': '缺少筛选阈值参数'}), 400

        # 打印列名以调试
        print("Output CSV列名:", output_df.columns.tolist())
        print("Decision Tree CSV列名:", decision_tree_df.columns.tolist())

        # 打印数据类型
        print("onset/initial temperature类型:", output_df['onset/initial dehydrogenation temperature(¡ãc)'].dtype)
        print("activation energy类型:", output_df['activation energy(ea)(kj/mol)'].dtype)

        # 将字符串类型的列转换为数值类型
        # 使用pd.to_numeric转换，errors='coerce'会将无法转换的值设为NaN
        output_df['onset/initial dehydrogenation temperature(¡ãc)'] = pd.to_numeric(
            output_df['onset/initial dehydrogenation temperature(¡ãc)'], errors='coerce')
        output_df['activation energy(ea)(kj/mol)'] = pd.to_numeric(
            output_df['activation energy(ea)(kj/mol)'], errors='coerce')

        # 在output_jd_merged.csv中执行筛选，使用dropna确保没有NaN值
        filtered_output_df = output_df.dropna(subset=[
            'onset/initial dehydrogenation temperature(¡ãc)',
            'activation energy(ea)(kj/mol)'
        ])[
            (output_df['onset/initial dehydrogenation temperature(¡ãc)'] < onset_temp_threshold) &
            (output_df['activation energy(ea)(kj/mol)'] < activation_threshold)
            ]

        if filtered_output_df.empty:
            return jsonify({'message': '未找到符合条件的记录'}), 404

        # 获取筛选后的催化剂组分列表
        catalyst_components = filtered_output_df['catalysts component'].unique().tolist()
        print("筛选出的催化剂组分:", catalyst_components)

        # 在Decision Tree数据集中查找这些催化剂组分
        matched_decision_tree_df = decision_tree_df[
            decision_tree_df['Catalysts_Component'].isin(catalyst_components)
        ]

        if matched_decision_tree_df.empty:
            return jsonify({'message': '在Decision Tree数据集中未找到匹配的催化剂组分'}), 404

        # 修改：确保推荐的三个材料具有不同的催化剂组分
        # 首先按Average_MAE升序排序
        sorted_df = matched_decision_tree_df.sort_values(by='Average_MAE', ascending=True)

        # 创建一个新的结果DataFrame
        result_df = pd.DataFrame(columns=sorted_df.columns)

        # 用于记录已选择的组分
        selected_components = set()

        # 从排序后的数据中选择不同组分的记录
        for _, row in sorted_df.iterrows():
            component = row['Catalysts_Component']
            if component not in selected_components:
                result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
                selected_components.add(component)
                # 如果已经选择了3个不同的组分，则结束循环
                if len(result_df) >= 3:
                    break

        # 如果选择的记录少于3个，返回相应的提示
        if len(result_df) < 3:
            print(f"Warning: 只找到 {len(result_df)} 个不同组分的记录")

        # 如果没有找到任何记录，返回错误
        if result_df.empty:
            return jsonify({'message': '无法找到具有不同组分的记录'}), 404

        # 返回所有字段数据
        result = result_df.to_dict(orient='records')
        return jsonify(result), 200

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("详细错误信息:", error_trace)
        return jsonify({'error': str(e), 'traceback': error_trace}), 500
if __name__ == '__main__':
    process_train("/home/hongchang/下载/MODEI_3/XGBoost_train_set.csv", perplexity=19)
    app.run(debug=True,port=3031)
