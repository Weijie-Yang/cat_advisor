import pandas as pd
import json
import openai
import httpx

# 定义文件路径
input_file = 'data/output_jd_deduplicated.csv'
output_json_file = 'data/output_jd_classified.json'

# 初始化OpenAI客户端
client = openai.OpenAI(
    base_url="",
    api_key="",  # 替换为你的API密钥
    http_client=httpx.Client(
        base_url="",
        follow_redirects=True,
    ),
)

system_content = """
You are a materials scientist specializing in the analysis and classification of hydrogen storage material catalysts, particularly for magnesium hydride (MgH2).

Your task is to categorize the following list of catalyst components into the specified categories:

The catalysts for magnesium-based hydrogen storage to catalyze MgH₂ mainly fall into the following categories:

1. **Transition Metal Elements**: Such as Ti, V, Nb, Fe, Co, Ni, Pd, etc. They possess excellent catalytic reaction activity and can generate intermetallic hydrides during the hydrogen absorption and desorption reactions, improving the kinetic performance of MgH₂. Among them, Ti has a relatively outstanding catalytic effect. For example, the hydrogen desorption kinetic performance of MgH₂-Ti is the most prominent. Ti can promote the dissociation of hydrogen molecules on its surface, thus accelerating the entry of hydrogen elements into the matrix. Meanwhile, it weakens the Mg—H bond, which is beneficial to the hydrogen desorption of the magnesium-based hydrogen storage system.
2. **Bimetallic or Multimetallic Alloys**: Such as alloy catalysts like Ni-Cu, Nb-Ti, Co/Pd, etc. They further improve the catalytic efficiency through the synergistic effect among different metals. For instance, the Co/Pd@BCNT catalyst with Co/Pd uniformly loaded on bamboo-shaped carbon nanotubes, during the hydrogen absorption process, the element Pd accelerates the diffusion of hydrogen at the interface between Pd and Mg atoms; while during the hydrogen desorption process, the mutual transformation between Mg₂Co and Mg₂CoH₅ effectively reduces the diffusion energy barrier and thus promotes the release of hydrogen atoms.
3. **Metal Oxides**: Such as TiO₂, Nb₂O₅, etc. They can provide hydrogen atom diffusion channels and weaken the Mg—H bond through the interaction with the surface of MgH₂, thereby improving the hydrogen absorption and desorption performance. For example, the graphene-like TiO₂(B) synthesized is reduced to metal Ti nanoparticles and wrinkled Ti₂O₃ during ball milling and (de)hydrogenation processes, forming many boundary interfaces between MgH₂ and Ti-based catalysts and facilitating the diffusion of hydrogen.
4. **Metal Carbides**: Such as NbC, TiC, etc. They have characteristics such as high hardness, high melting point and good chemical stability. They can serve as effective catalysts in the MgH₂ hydrogen storage system and improve the hydrogen absorption and desorption kinetic performance.
5. **Metal Nitrides**: Such as VN, TiN, etc. There are a large number of vacancies and defects in their crystal structures, and these vacancies and defects can serve as adsorption and diffusion sites for hydrogen atoms, thus improving the hydrogen absorption and desorption performance of MgH₂.
6. **Metal Halides**: Such as TiCl₃, etc. They generate multivalent metal phases through in-situ reactions and jointly act as intermediate catalysts for electron transfer to promote the decomposition of MgH₂ and significantly reduce the dehydrogenation temperature of MgH₂.
7. **Carbon Materials**: Such as activated carbon, carbon nanotubes, graphene, etc. They have a large specific surface area and good electrical conductivity. They can be used as catalyst carriers and can also directly participate in catalytic reactions to improve the hydrogen absorption and desorption performance of MgH₂. For example, the composite catalyst of nickel-copper alloy loaded on reduced graphene oxide prevents the agglomeration of metal particles and improves the activity and stability of the catalyst.
8. **Borides**: Such as TiB₂, ZrB₂, etc. They have characteristics such as high melting point, high hardness and good chemical stability. They can serve as effective catalysts in the MgH₂ hydrogen storage system and improve the hydrogen absorption and desorption kinetic performance.
9. **Nitrides**: Such as BN, etc. They have good chemical stability and thermal stability and can serve as effective catalysts in the MgH₂ hydrogen storage system and improve the hydrogen absorption and desorption kinetic performance.
10.**Others**:For example, some catalysts with radioactive elements as the basic catalytic phase, such as the (dmf)₂((uo₂)₂(h₂o)₁.₅l₄⁻) crystal material prepared by the hydrothermal reaction, also have a good catalytic effect on MgH₂. 
11.**Metal and carbon-based composite Catalysts**
12.**Composite catalyst**
Please classify the following list of catalyst components and output the results in JSON format. 

**Important Notes:**

*   Ignore any percentages listed after the component name (e.g., "99%").
*   "锰" is the Chinese word for Manganese (Mn).
*   Treat components with "@" or "/rGO" as Carbon-Based Catalysts.

```json
{
Catalyst category:Metal Halides
}

"""

def classify_catalyst(component):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "system", "content": system_content}, {"role": "user", "content": component}]
        )
        response_content = response.choices[0].message.content
        return response_content.strip()
    except Exception as e:
        print(f"Error classifying catalyst {component}: {e}")
        return "Unknown"

# 读取CSV文件
data = pd.read_csv(input_file)

# 创建一个列表来保存所有JSON对象
json_objects = []

for index, row in data.iterrows():
    catalyst_component = row['catalysts component']
    year = row['year']
    journal_title = row['Journal Title']
    title = row['title']

    # 对催化剂成分进行分类
    category = classify_catalyst(catalyst_component)

    # 构建JSON对象
    json_object = {
        "catalysts component": catalyst_component,
        "Catalyst category": category,
        "year": year,
        "Journal Title": journal_title,
        "title": title
    }

    json_objects.append(json_object)

# 将所有JSON对象保存到文件
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(json_objects, f, ensure_ascii=False, indent=4)

print(f"分类结果已成功保存至 {output_json_file}")