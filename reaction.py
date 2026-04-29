from flask import Flask, jsonify, render_template, request
from rdkit import Chem
from rdkit.Chem import rdChemReactions

app = Flask(__name__)

# RDKit動態推演設定 (動態化學引擎)
DYNAMIC_RULES = [
    {
        "name" : "酯化反應(Esterification)",
        "rxn" : rdChemReactions.ReactionFromSmarts("[CX3:1](=[OX1:2])[OX2H1:3].[OX2H1:4][CX4:5]>>[CX3:1](=[OX1:2])[OX2:4][CX4:5].O"),
        "energy" : "微放熱(Mildly Exothermic)",
        "warning" : "產物具有特殊香味"
    },
    {
        "name" : "醯胺合成 (Amide Formation)",
        "rxn" : rdChemReactions.ReactionFromSmarts("[CX3:1](=[OX1:2])[OX2H1:3].[NX3H2:4][C:5]>>[CX3:1](=[OX1:2])[NX3H1:4][C:5].O"),
        "energy" : "放熱 (Exothermic)",
        "warning" : "生成穩定的醯胺鍵結"
    }
]

# SMILES翻譯對照表
SMILES_MAP = {
    "Acetic Acid" : "CC(=O)O",              #乙酸
    "Ethanol" : "CCO",                      #乙醇
    "Methylamine" : "CN",                   #甲胺
    "Na" : "[Na]",
    "H2O" : "O",
    "HCl" : "Cl",
    "NaOH" : "[Na+].[OH-]",
    "Fe" : "[Fe]",                          #鐵
    "CuSO4" : "[Cu+2].[O-]S(=O)(=O)[O-]",   #硫酸銅
    "NaHCO3" : "[Na+].OC(=O)[O-]",           #碳酸氫鈉(小蘇打)
    "CaCO3": "[Ca+2].[O-]C(=O)[O-]",
    "NaCl": "[Na+].[Cl-]",
    "H2": "[H][H]",
    "CO2": "O=C=O",
    "CaCl2": "[Ca+2].[Cl-].[Cl-]"
}

# 靜態反應資料庫 (無機物與特殊反應的Fallback)
REACTION_DB = {
    frozenset(["AgNO3", "NaCl"]): {
        "equation": "AgNO₃ + NaCl → AgCl ↓ + NaNO₃",
        "type": "沉澱反應 (Precipitation)",
        "energy": "低能量 (Low Energy)",
        "warning": "產生白色固體沉澱"
    },
    frozenset(["Fe", "CuSO4"]): {
        "equation": "Fe + CuSO₄ → FeSO₄ + Cu ↓",
        "type": "單置換反應 (Single Displacement)",
        "energy": "低能量 (Low Energy)",
        "warning": "鐵表面會析出紅色的銅金屬"
    },
    frozenset(["NaHCO3", "Acetic Acid"]): {
        "equation": "NaHCO₃ + CH₃COOH → CH₃COONa + H₂O + CO₂ ↑",
        "type": "酸鹼發泡反應 (Gas Evolution)",
        "energy": "吸熱 (Endothermic)",
        "warning": "產生大量二氧化碳氣泡，溫度下降"
    },
    frozenset(["Acetic Acid", "Ethanol"]): {
        "equation": "CH₃COOH + C₂H₅OH ⇌ CH₃COOC₂H₅ + H₂O",
        "type": "酯化反應 (Esterification)",
        "energy": "微放熱 (Mildly Exothermic)",
        "warning": "實際操作需加入濃硫酸作為催化劑"
    },
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
    },
    frozenset(["CaCO3", "HCl"]): {
        "equation": "CaCO₃ + 2HCl → CaCl₂ + H₂O + CO₂ ↑",
        "type": "酸鹼反應", "energy": "放熱", "warning": "產生二氧化碳氣體",
        "products": [
            {"name": "氯化鈣 (CaCl2)", "value": "CaCl2"},
            {"name": "水 (H2O)", "value": "H2O"},
            {"name": "二氧化碳 (CO2)", "value": "CO2"}
        ]
    }
}

SAVED_EXPERIMENTS = []

# 輔助函數
def list_available_reactions():
    """供前端左側選單渲染使用"""
    return [
        {"id": idx, "label": " + ".join(key)}
        for idx, key in enumerate(REACTION_DB.keys(), start=1)
    ]

def get_reaction_by_index(index):
    """供前端點擊左側範例時呼叫"""
    keys = list(REACTION_DB.keys())
    if 1 <= index <= len(keys):
        selected_key = keys[index - 1]
        data = REACTION_DB[selected_key]
        return {
            "chemicals": list(selected_key),
            "equation": data["equation"],
            "type": data["type"],
            "energy": data["energy"],
            "warning": data["warning"],
            "success": True
        }
    return {"success": False}

# API route
@app.route('/')
def index():
    reactions = list_available_reactions()
    return render_template('index.html', reactions=reactions, saved=SAVED_EXPERIMENTS)

@app.route('/api/react/<int:reaction_id>')
def trigger_reaction(reaction_id):
    result = get_reaction_by_index(reaction_id)
    return jsonify(result)

@app.route('/api/solve', methods=['POST'])
def solve_reaction():
    data = request.json
    inputs = data.get('chemicals', [])
    
    # 物理引擎通常是一對一碰撞，因此預期陣列長度為2
    if len(inputs) != 2:
        return jsonify({"success": False, "message": "目前動態僅支援雙物件反應"})

    #將前端名稱轉換為 SMILES 結構
    smiles_1 = SMILES_MAP.get(inputs[0], inputs[0])
    smiles_2 = SMILES_MAP.get(inputs[1], inputs[1])
    
    mol1 = Chem.MolFromSmiles(smiles_1)
    mol2 = Chem.MolFromSmiles(smiles_2)

    if mol1 and mol2:
        for rule in DYNAMIC_RULES:
            try:
                products = rule["rxn"].RunReactants((mol1, mol2))
            #若無反應，對調反應物順序再試一次
                if not products:
                    products = rule["rxn"].RunReactions((mol2, mol1))
                
                if products:
                    product_list = []
                    for idx, prod_mol in enumerate(products[0]):
                        prod_smiles = Chem.MolToSmiles(prod_mol)
                        prod_name = f"有機產物 ({prod_smiles})" if idx == 0 else f"副產物 ({prod_smiles})"
                        product_list.append({"name": prod_name, "value": prod_smiles})
                    
                    return jsonify({
                        "success": True,
                        "equation": f"{smiles_1} + {smiles_2} → " + " + ".join([p["value"] for p in product_list]),
                        "type": rule["name"],
                        "energy": rule["energy"],
                        "warning": rule["warning"],
                        "products": product_list,
                        "is_dynamic": True
                    })
            except Exception as e:
                print(f"Rule {rule['name']} Failed: {e}")

    #Fallback至靜態資料庫
    input_set = set(inputs)
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
            
    #如果RDKit算不出來，且資料庫也沒有，則視為無反應
    return jsonify({"success": False, "message": "無化學反應發生，或尚未定義此規則"})
    
@app.route('/api/save', methods=['POST'])
def save_experiment():
    data = request.json
    new_record = {
        "id" : len(SAVED_EXPERIMENTS) + 1,
        "label" : data.get("equation", "Unknown Reaction"),
        "type" : data.get("type", "Unknown")
    }
    SAVED_EXPERIMENTS.append(new_record)
    return jsonify({"success": True, "record": new_record})

if __name__ == '__main__':
    app.run(debug=True)