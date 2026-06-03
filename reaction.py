import json
import os
import requests
from flask import Flask, jsonify, render_template, request
# pyrefly: ignore [missing-import]
from rdkit import Chem
# pyrefly: ignore [missing-import]
from rdkit.Chem import rdChemReactions
# pyrefly: ignore [missing-import]
from rdkit import RDLogger
# 停用 RDKit 的解析錯誤與警告輸出
RDLogger.DisableLog('rdApp.*')
from local_engine import LocalReactionPredictor

app = Flask(__name__)
predictor = LocalReactionPredictor()

PERIODIC_TABLE_FILE = 'periodic_table.json'
periodic_table_data = []
if os.path.exists(PERIODIC_TABLE_FILE):
    with open(PERIODIC_TABLE_FILE, 'r', encoding='utf-8') as f:
        try:
            periodic_table_data = json.load(f)
        except Exception as e:
            print(f"Error loading periodic table: {e}")

# DYNAMIC_RULES 已經由 local_engine 內建的 ML 模板取代，因此移除手寫規則

CUSTOM_CHEM_FILE = 'custom_chemicals.json'
custom_chems = {}

if os.path.exists(CUSTOM_CHEM_FILE):
    with open(CUSTOM_CHEM_FILE, 'r', encoding='utf-8') as f:
        try:
            custom_chems = json.load(f)
        except Exception as e:
            print(f"Error loading custom chemicals: {e}")

def get_smiles_from_name(name):
    """
    自動化學名稱解析器：
    1. 呼叫 PubChem API 查詢 (優先使用，確保 SMILES 正確性)
    2. 測試原始字串是否本來就是合法 SMILES
    3. 查閱本機 custom_chemicals
    4. 測試是否為無機元素符號 (加上括號)
    """
    if not name: return None

    #查閱本機 custom_chemicals
    if name in custom_chems:
        return custom_chems[name]

    #呼叫 PubChem API
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES/JSON"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            props = data['PropertyTable']['Properties'][0]
            smiles = props.get('IsomericSMILES') or props.get('CanonicalSMILES') or props.get('SMILES')
            if smiles:
                return smiles
            else:
                print(f"[PubChem API] 回傳資料中不包含 SMILES: {props}")
        else:
            print(f"[PubChem API] 找不到 '{name}' 或請求失敗，狀態碼: {res.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[PubChem API] 網路請求 '{name}' 發生錯誤: {e}")
    except Exception as e:
        print(f"[PubChem API] 解析 '{name}' 資料時發生未知的錯誤: {e}")

    #測試原始字串是否本來就是合法SMILES (例如有機子集或使用者手動輸入的 SMILES)
    mol = Chem.MolFromSmiles(name)
    if mol: return name
    
    return None

@app.route('/')
def index():
    element_symbols = [el['symbol'] for el in periodic_table_data] if periodic_table_data else []
    return render_template('index.html', custom_chems=custom_chems, element_symbols=element_symbols)

@app.route('/api/elements', methods=['GET'])
def get_elements():
    return jsonify(periodic_table_data)

@app.route('/api/add_chemical', methods=['POST'])
def add_chemical():
    data = request.json
    name = data.get('name', '').strip()
    smiles = data.get('smiles', '').strip()

    # 若使用者沒填寫 SMILES，我們嘗試自動從 API 抓取
    if not smiles:
        smiles = get_smiles_from_name(name)
        if not smiles:
            return jsonify({"success": False, "message": f"無法從公開資料庫自動找到 '{name}' 的 SMILES，請手動輸入。"})

    # 使用 RDKit 驗證 SMILES 是否合法
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return jsonify({"success": False, "message": f"解析失敗，'{smiles}' 不是合法的 SMILES！"})

    # 驗證成功，存入 JSON 檔案中
    custom_chems[name] = smiles

    with open(CUSTOM_CHEM_FILE, 'w', encoding='utf-8') as f:
        json.dump(custom_chems, f, ensure_ascii=False, indent=4)

    return jsonify({"success": True, "name": name, "smiles": smiles})

@app.route('/api/solve', methods=['POST'])
def solve_reaction():
    data = request.json
    inputs = data.get('chemicals', [])
    
    if len(inputs) != 2:
        return jsonify({"success": False, "message": "目前動態引擎僅支援雙物種反應"})

    # 透過自動解析器將名稱轉換為 SMILES
    smiles_1 = get_smiles_from_name(inputs[0])
    smiles_2 = get_smiles_from_name(inputs[1])
    
    if not smiles_1 or not smiles_2:
        failed_name = inputs[0] if not smiles_1 else inputs[1]
        return jsonify({"success": False, "message": f"系統無法辨識 '{failed_name}' 的化學結構，請先透過左側選單將其加入靜態資料庫。"})
    
    mol1 = Chem.MolFromSmiles(smiles_1)
    mol2 = Chem.MolFromSmiles(smiles_2)

    # 使用 A+B 本機 ML 預測引擎進行預測
    # 此作法取代了原先人工寫死的 DYNAMIC_RULES
    if mol1 and mol2:
        prediction_result = predictor.predict([smiles_1, smiles_2])
        if prediction_result["success"]:
            product_list = []
            for idx, prod_smiles in enumerate(prediction_result["products"]):
                prod_name = f"{prod_smiles} (主產物)" if idx == 0 else f"{prod_smiles} (副產物)"
                product_list.append({"name": prod_name, "value": prod_smiles})

            equation_str = f"{smiles_1} + {smiles_2} → " + " + ".join([p["value"] for p in product_list])
            return jsonify({
                "success": True,
                "equation": equation_str,
                "type": f"AI/ML 預測路徑 ({prediction_result['template_used']})",
                "energy": "依反應不同",
                "warning": "由本機模型推導生成",
                "products": product_list,
                "is_dynamic": True
            })

    return jsonify({"success": False, "message": "ML 引擎未能預測出合理的化學反應。"})

if __name__ == '__main__':
    app.run(debug=True)
