import os
import json
import openai
import httpx
import re

# 初始化OpenAI客户端
client = openai.OpenAI(
    base_url="",
    api_key="",  # 替换为你的API密钥
    http_client=httpx.Client(
        base_url="",
        follow_redirects=True,
    ),
)

def clean_invalid_escapes(raw_text):
    """清理无效的转义字符并修复 JSON 格式"""
    # 移除可能的 LaTeX 表达式的 "$" 标记
    cleaned_text = re.sub(r'\$\{\\mathrm\{(.*?)\}\}\$', r'\1', raw_text)
    
    # 去除 ```json 和 ``` 等标记
    cleaned_text = re.sub(r'^```json\n', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\n```$', '', cleaned_text, flags=re.MULTILINE)
    
    # 替换 Unicode 字符
    cleaned_text = cleaned_text.replace("\u2103", "°C").replace("\uff08", "(").replace("\uff09", ")")
    
    # 确保 JSON 开始和结束符号正确
    if not cleaned_text.startswith('{'):
        cleaned_text = '{' + cleaned_text
    if not cleaned_text.endswith('}'):
        cleaned_text = cleaned_text + '}'
    
    # 去除可能的多余逗号
    cleaned_text = re.sub(r',\s*}', '}', cleaned_text)
    
    return cleaned_text

def process_md_file(file_path, output_folder):
    """处理单个MD文件并使用GPT提取信息"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    system_content = """
You are a data extraction assistant. Please read the provided paper carefully and extract the information for the following fields. If a certain field's information is not found, please fill in "NA".
For fields that can have multiple values (such as "Catalytic Mass Fraction(wt%)" and "Catalysts Particle Size"), please **split** each scenario into **separate JSON objects**, one for each value. For example, if a field contains multiple values (e.g., "1-2 µm, 3-4 µm"), return them as separate JSON objects in separate entries. Each value should appear in its own entry with other fields matching the original row.
Please note that **"onset dehydrogenation temperature(℃)"** and **"initial dehydrogenation temperature(℃)"** refer to the same concept. If either field is found, please combine them as a single field.

The fields that need to be extracted include:
- Name of Alloy
- hydrogen desorption Catalysts Particle Size [If multiple values are listed, create separate entries for each value]
- Catalysts Component
- Catalytic Mass Fraction(wt%) [If multiple values are listed, create separate entries for each value]
- Ball Milling Mass Ratio
- Ball Milling Rotating Speed(rpm)
- Ball Milling Time(min)
- hydrogen desorption PCT Plateau Pressure(bar)
- hydrogen desorption PCT Temperature(℃)
- Desorption Performance PCT ΔH(kJ/mol)
- Desorption Performance PCT ΔS(J/mol/K)
- onset/initial dehydrogenation temperature(℃) [Combine both "onset dehydrogenation temperature(℃)" and "initial dehydrogenation temperature(℃)" as one field]
- TPD, TG Maximum Capacity(wt%)
- TPD, TG Pressure(MPa)
- Activation Energy(Ea)(kJ/mol)
- Kinetics Pressure(MPa)
- Kinetics Temperature(℃)
- Cycle Performance Cycles
- Cycle Performance Loss Per Cycle(wt%)

Please ensure the output format is in **JSON**. For each table entry, if a field contains multiple values, return a separate JSON object for each scenario. If any field is not mentioned in the text, it should still be included with its value set to "NA".

"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "system", "content": system_content}, {"role": "user", "content": content}]
        )
        # 获取返回的文本内容
        response_content = response.choices[0].message.content
        
        # Log the raw response for debugging
        print("Raw GPT Response:", response_content)
        
        # 清理无效的转义字符并修复 JSON 格式
        cleaned_content = clean_invalid_escapes(response_content)
        
        # 打印清理后的内容
        print("Cleaned GPT Response:", cleaned_content)
        
        # Save cleaned content to a .txt file instead of JSON
        output_txt_path = os.path.join(output_folder, os.path.basename(file_path) + "_cleaned.txt")
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(cleaned_content)
        
        print(f"Cleaned content saved to {output_txt_path}")
        
    except Exception as e:
        print(f"Error processing file with GPT: {str(e)}")

def process_folder(folder_path, output_folder):
    """处理文件夹中的所有MD文件"""
    # 获取所有MD文件
    md_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
    
    for md_file in md_files:
        file_path = os.path.join(folder_path, md_file)
        try:
            print(f"Processing {md_file}...")
            process_md_file(file_path, output_folder)
        except Exception as e:
            print(f"Error processing {md_file}: {str(e)}")

if __name__ == "__main__":
    # 替换为你的文件夹路径
    folder_path = "data\\md\\test2"
    output_folder = "data\\c_txt\\test2_data"  # 设置输出文件夹路径
    process_folder(folder_path, output_folder)
