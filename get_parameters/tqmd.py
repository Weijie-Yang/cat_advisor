import os
import json
import openai
import httpx
import re
import logging

# --- 配置 ---
# 初始化OpenAI客户端
client = openai.OpenAI(
    base_url="YOUR_BASE_URL",  # 替换为你的API基础URL
    api_key="YOUR_API_KEY",    # 替换为你的API密钥
    http_client=httpx.Client(
        base_url="YOUR_BASE_URL",
        follow_redirects=True,
    ),
)
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
#  数据清洗与验证函数
# ==============================================================================
def extract_json_from_llm_response(raw_text):
    """从LLM的原始响应中稳健地提取JSON内容。"""
    # 查找被 ```json ... ``` 包围的代码块
    match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_text)
    if match:
        return match.group(1).strip()
    
    # 如果没有找到代码块，尝试找到第一个 '{' 和最后一个 '}' 之间的内容
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return raw_text[start:end+1].strip()
        
    logging.warning("Could not find a valid JSON block in the LLM response.")
    return None

def perform_sanity_checks_and_corrections(data_list):
    """
    对提取出的数据执行合理性检查和自动修正。
    这是实现你回复信中承诺的核心功能。
    """
    corrected_list = []
    if not isinstance(data_list, list):
        data_list = [data_list] # 确保输入是列表

    for item in data_list:
        if not isinstance(item, dict): continue

        # --- 检查和修正温度单位 ---
        temp_col = 'onset/initial dehydrogenation temperature(℃)'
        if temp_col in item and isinstance(item[temp_col], (int, float)):
            temp_value = item[temp_col]
            # 规则：如果温度 > 500，假定是开尔文，并转换为摄氏度
            if temp_value > 500:
                corrected_temp = temp_value - 273.15
                logging.info(f"Temperature unit correction: {temp_value}K -> {corrected_temp:.2f}°C")
                item[temp_col] = round(corrected_temp, 2)
        
        # --- 可以在这里添加更多规则 ---
        # 例如：检查活化能是否在合理范围
        ea_col = 'Activation Energy(Ea)(kJ/mol)'
        if ea_col in item and isinstance(item[ea_col], (int, float)):
            if item[ea_col] < 0 or item[ea_col] > 500:
                logging.warning(f"Unusual Activation Energy value found: {item[ea_col]} kJ/mol")

        corrected_list.append(item)
    return corrected_list

# ==============================================================================
#  主处理流程
# ==============================================================================
def process_md_file(file_path, output_folder):
    """处理单个MD文件并使用GPT提取、清洗和验证信息"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    system_prompt = """
    You are a data extraction assistant specializing in materials science. Please read the provided paper carefully and extract the information for the following 16 predefined experimental parameters.
    - Your output **MUST** be a valid JSON object or a list of JSON objects. Do not include any explanatory text before or after the JSON.
    - If a parameter has multiple distinct values corresponding to different experimental conditions (e.g., different mass fractions, particle sizes), create a **separate JSON object for each condition**.
    - If a parameter is not mentioned, include the key in the JSON with a value of "NA".
    - Combine "onset dehydrogenation temperature(℃)" and "initial dehydrogenation temperature(℃)" into a single field: "onset/initial dehydrogenation temperature(℃)".
    
    Fields to extract:
    [Name of Alloy, hydrogen desorption Catalysts Particle Size, Catalysts Component, Catalytic Mass Fraction(wt%), Ball Milling Mass Ratio, Ball Milling Rotating Speed(rpm), Ball Milling Time(min), hydrogen desorption PCT Plateau Pressure(bar), hydrogen desorption PCT Temperature(℃), Desorption Performance PCT ΔH(kJ/mol), Desorption Performance PCT ΔS(J/mol/K), onset/initial dehydrogenation temperature(℃), TPD, TG Maximum Capacity(wt%), TPD, TG Pressure(MPa), Activation Energy(Ea)(kJ/mol), Kinetics Pressure(MPa), Kinetics Temperature(℃), Cycle Performance Cycles, Cycle Performance Loss Per Cycle(wt%)]
    """
    
    try:
        logging.info(f"Sending request to GPT-4o for {os.path.basename(file_path)}...")
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
            response_format={"type": "json_object"} # 使用JSON模式以获得更可靠的输出
        )
        response_content = response.choices[0].message.content
        
        # 从响应中稳健地提取JSON部分
        json_str = extract_json_from_llm_response(response_content)
        if not json_str:
            raise ValueError("Failed to extract JSON from LLM response.")
            
        # 解析JSON
        data = json.loads(json_str)
        
        # 执行合理性检查和修正
        corrected_data = perform_sanity_checks_and_corrections(data)
        
        # 保存为结构化的JSON文件
        output_json_path = os.path.join(output_folder, os.path.basename(file_path).replace('.md', '.json'))
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(corrected_data, f, indent=4, ensure_ascii=False)
        
        logging.info(f"Successfully processed and saved to {output_json_path}")
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON Decode Error for {file_path}: {e}. Raw response was: {response_content}")
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}", exc_info=True)

def process_folder(folder_path, output_folder):
    """处理文件夹中的所有MD文件"""
    ensure_directory_exists(output_folder)
    for md_file in os.listdir(folder_path):
        if md_file.endswith('.md'):
            file_path = os.path.join(folder_path, md_file)
            process_md_file(file_path, output_folder)

if __name__ == "__main__":
    folder_path = "data/md/test2"
    output_folder = "data/json_processed/test2_data"
    process_folder(folder_path, output_folder)
