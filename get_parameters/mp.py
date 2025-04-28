import json
from mp_api.client import MPRester

# 你的Materials Project API密钥
api_key = ''

# 输入文件路径
input_file_path = "catalysts_component2.txt"

# 输出文件路径
output_file_path = "materials_data.json"

# 从文本文件中读取化学式列表
with open(input_file_path, 'r') as file:
    formulas = [line.strip() for line in file if line.strip()]  # 去除空行和空白字符

# 使用你的API密钥初始化MPRester
with MPRester(api_key=api_key) as mpr:
    with open(output_file_path, 'w') as file:
        file.write('[')  # 开始JSON数组
        first_entry = True  # 标记是否是第一个条目

        for formula in formulas:
            try:
                # 搜索指定化学式的材料
                search_results = mpr.summary.search(formula=formula)
                
                # 如果没有找到匹配项，则跳过
                if not search_results:
                    print(f"未找到 {formula} 的相关信息。")
                    continue
                
                for result in search_results:
                    material_id = result.material_id
                    material_data_summary = mpr.summary.get_data_by_id(material_id)
                    material_data_structure = mpr.materials.get_structure_by_material_id(material_id)

                    # 构建材料数据字典
                    material_info = {
                        "Material ID": material_id,
                        "Formula": result.formula_pretty,
                        "Formula Anonymous": material_data_summary.formula_anonymous,
                        "Is Gap Direct": material_data_summary.is_gap_direct,
                        "CBM": material_data_summary.cbm,
                        "VBM": material_data_summary.vbm,
                        "Density (g/cm^3)": material_data_summary.density,
                        "Energy per Atom (eV)": material_data_summary.energy_per_atom,
                        "Volume (cm^3)": material_data_summary.volume,
                        "Band Gap (eV)": material_data_summary.band_gap,
                        "Formation Energy (eV/atom)": material_data_summary.formation_energy_per_atom,
                        "Fermi Energy (eV)": material_data_summary.efermi,
                        "Electronic Energy (eV)": material_data_summary.e_electronic,
                        "Total Energy (eV)": material_data_summary.e_total,
                        "Ionic Energy (eV)": material_data_summary.e_ionic,
                        "Equilibrium Reaction Energy per Atom (eV/atom)": material_data_summary.equilibrium_reaction_energy_per_atom,
                        "Elements": [str(element) for element in material_data_summary.elements],
                        "Grain Boundaries": getattr(material_data_summary, "grain_boundaries", None),
                        "Surface Anisotropy (J/m^2)": material_data_summary.surface_anisotropy,
                        "Weighted Surface Energy": material_data_summary.weighted_surface_energy,
                        "Lattice Parameters": {
                            "a": material_data_structure.lattice.a,
                            "b": material_data_structure.lattice.b,
                            "c": material_data_structure.lattice.c,
                            "alpha": material_data_structure.lattice.alpha,
                            "beta": material_data_structure.lattice.beta,
                            "gamma": material_data_structure.lattice.gamma
                        }
                    }

                    # 写入JSON文件
                    if not first_entry:
                        file.write(',\n')
                    else:
                        first_entry = False
                    json.dump(material_info, file, indent=4, ensure_ascii=False)
                    
            except Exception as e:
                print(f"处理 {formula} 时发生错误: {e}")

        file.write('\n]')  # 结束JSON数组

print(f"数据已成功导出到: {output_file_path}")