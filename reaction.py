import json
import os
from flask import Flask, jsonify, render_template, request
# pyrefly: ignore [missing-import]
from rdkit import Chem
# pyrefly: ignore [missing-import]
from rdkit.Chem import rdChemReactions

app = Flask(__name__)

PERIODIC_TABLE_FILE = 'periodic_table.json'
periodic_table_data = []
if os.path.exists(PERIODIC_TABLE_FILE):
    with open(PERIODIC_TABLE_FILE, 'r', encoding='utf-8') as f:
        try:
            periodic_table_data = json.load(f)
        except Exception as e:
            print(f"Error loading periodic table: {e}")

DYNAMIC_RULES = [
    {
        "name": "酯化反應 (Esterification)",
        "rxn": rdChemReactions.ReactionFromSmarts("[CX3:1](=[OX1:2])[OX2H1:3].[OX2H1:4][CX4:5]>>[CX3:1](=[OX1:2])[OX2:4][CX4:5].O"),
        "energy": "微放熱",
        "warning": "產物通常具有水果香味"
    },
    {
        "name": "醯胺合成 (Amide Formation)",
        "rxn": rdChemReactions.ReactionFromSmarts("[CX3:1](=[OX1:2])[OX2H1:3].[NX3H2:4][C:5]>>[CX3:1](=[OX1:2])[NX3H1:4][C:5].O"),
        "energy": "放熱",
        "warning": "生成穩定的醯胺鍵結"
    },
    {
        "name": "縮醛反應 (Acetal Formation)",
        "rxn": rdChemReactions.ReactionFromSmarts("[CX3:1](=[OX1:2])[C:3].[OX2H1:4][CX4:5]>>[C:3][C:1]([OX2:2][CX4:5])([OX2:4][CX4:5]).O"),
        "energy": "微吸熱",
        "warning": "需要酸性催化"
    }
]

SMILES_MAP = {
    "Acetic Acid": "CC(=O)O", 
    "Ethanol": "CCO",         
    "Methylamine": "CN",      
    "Na": "[Na]",
    "H2O": "O",
    "HCl": "Cl",
    "NaOH": "[Na+].[OH-]",
    "Fe": "[Fe]",             
    "CuSO4": "[Cu+2].[O-]S(=O)(=O)[O-]", 
    "NaHCO3": "[Na+].OC(=O)[O-]",
    "CaCO3": "[Ca+2].[O-]C(=O)[O-]"
}

REACTION_DB = {
    frozenset(["Na", "H2O"]): {
        "equation": "2Na + 2H₂O → 2NaOH + H₂ ↑",
        "type": "劇烈反應", "energy": "高度放熱", "warning": "警告：產生易燃氫氣",
        "products": [
            {"name": "氫氧化鈉 (NaOH)", "value": "NaOH"},
            {"name": "氫氣 (H2)", "value": "H2"}
        ]
    },
    frozenset(["HCl", "NaOH"]): {
        "equation": "HCl + NaOH → NaCl + H₂O",
        "type": "酸鹼中和", "energy": "放熱", "warning": "安全",
        "products": [
            {"name": "氯化鈉 (NaCl)", "value": "NaCl"},
            {"name": "水 (H2O)", "value": "H2O"}
        ]
    }
}

BASIC_BONDING_DB = {
    frozenset(["Na", "Cl"]): {"formula": "NaCl", "name": "氯化鈉"},
    frozenset(["H", "O"]): {"formula": "H2O", "name": "水"},
    frozenset(["C", "O"]): {"formula": "CO2", "name": "二氧化碳"},
    frozenset(["H", "Cl"]): {"formula": "HCl", "name": "氯化氫"},
    frozenset(["Fe", "O"]): {"formula": "Fe2O3", "name": "氧化鐵"},
    frozenset(["Na", "O"]): {"formula": "Na2O", "name": "氧化鈉"},
    frozenset(["Mg", "O"]): {"formula": "MgO", "name": "氧化鎂"},
    frozenset(["Ca", "O"]): {"formula": "CaO", "name": "氧化鈣"},
    frozenset(["K", "Cl"]): {"formula": "KCl", "name": "氯化鉀"}
}

CUSTOM_CHEM_FILE = 'custom_chemicals.json'
custom_chems = {}

if os.path.exists(CUSTOM_CHEM_FILE):
    with open(CUSTOM_CHEM_FILE, 'r', encoding='utf-8') as f:
        try:
            custom_chems = json.load(f)
            for name, smiles in custom_chems.items():
                SMILES_MAP[name] = smiles
        except Exception as e:
            print(f"Error loading custom chemicals: {e}")

@app.route('/')
def index():
    return render_template('index.html', custom_chems=custom_chems)

@app.route('/api/elements', methods=['GET'])
def get_elements():
    return jsonify(periodic_table_data)

@app.route('/api/add_chemical', methods=['POST'])
def add_chemical():
    data = request.json
    name = data.get('name', '').strip()
    smiles = data.get('smiles', '').strip()

    if not name or not smiles:
        return jsonify({"success": False, "message": "名稱與 SMILES 不能為空"})

    # 使用 RDKit 驗證輸入的 SMILES 是否合法
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return jsonify({"success": False, "message": "RDKit 無法解析此 SMILES，請檢查格式是否正確！"})

    # 驗證成功，存入記憶體與 JSON 檔案中
    SMILES_MAP[name] = smiles
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

    smiles_1 = SMILES_MAP.get(inputs[0], inputs[0])
    smiles_2 = SMILES_MAP.get(inputs[1], inputs[1])
    
    mol1 = Chem.MolFromSmiles(smiles_1)
    mol2 = Chem.MolFromSmiles(smiles_2)

    if mol1 and mol2:
        for rule in DYNAMIC_RULES:
            try:
                products = rule["rxn"].RunReactants((mol1, mol2))
                if not products:
                    products = rule["rxn"].RunReactants((mol2, mol1))
                    
                if products:
                    product_list = []
                    for idx, prod_mol in enumerate(products[0]):
                        prod_smiles = Chem.MolToSmiles(prod_mol)
                        if idx == 0:
                            prod_name = f"{prod_smiles} (主產物)"
                        else:
                            prod_name = f"{prod_smiles} (副產物)"
                        
                        product_list.append({"name": prod_name, "value": prod_smiles})

                    equation_str = f"{smiles_1} + {smiles_2} → " + " + ".join([p["value"] for p in product_list])
                    return jsonify({
                        "success": True,
                        "equation": equation_str,
                        "type": rule["name"],
                        "energy": rule["energy"],
                        "warning": rule["warning"],
                        "products": product_list,
                        "is_dynamic": True
                    })
            except Exception as e:
                print(f"RDKit Rule {rule['name']} Failed: {e}")

    input_set = set(inputs)
    
    # Check basic chemical bonding first
    for key, val in BASIC_BONDING_DB.items():
        if key == input_set:
            return jsonify({
                "success": True,
                "equation": f"{list(input_set)[0]} + {list(input_set)[1]} → {val['formula']}",
                "type": "基礎鍵結",
                "energy": "鍵結形成 (通常放熱)",
                "warning": "元素直接化合生成簡單分子",
                "products": [{"name": val["name"], "value": val["formula"]}],
                "is_dynamic": False
            })
    for key, val in REACTION_DB.items():
        if key == input_set:
            return jsonify({
                "success": True,
                "equation": val["equation"],
                "type": val["type"],
                "energy": val["energy"],
                "warning": val["warning"],
                "products": val.get("products", []),
                "is_dynamic": False
            })
            
    return jsonify({"success": False, "message": "無化學反應發生，或尚未定義此規則"})

if __name__ == '__main__':
    app.run(debug=True)
