import json
import os
import numpy as np
# pyrefly: ignore [missing-import]
from rdkit import Chem
# pyrefly: ignore [missing-import]
from rdkit.Chem import AllChem
# pyrefly: ignore [missing-import]
from rdkit.Chem import rdChemReactions
# pyrefly: ignore [missing-import]
from rdkit import DataStructs
try:
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    RandomForestClassifier = None

# 隱藏的基礎化學反應模板庫 (涵蓋常見的反應核心)
HIDDEN_TEMPLATES = [
    {"id": "T_HALOGENATION", "name": "鹵素化反應 (Halogenation)", "smarts": "[H:1][H:2].[F,Cl,Br,I:3][F,Cl,Br,I:4]>>[*:3][H:1].[*:4][H:2]"},
    {"id": "T_WATER", "name": "氫氧化反應 (Water Formation)", "smarts": "[H:1][H:2].[O:3]=[O:4]>>[H:1][O:3][H:2]"},
    {"id": "T_ELEMENT_OXIDATION", "name": "元素燃燒/氧化 (Element Oxidation)", "smarts": "[*:1].[O:2]=[O:3]>>[O:2]=[*:1]=[O:3]"},
    {"id": "T_ESTERIFICATION", "name": "酯化反應 (Esterification)", "smarts": "[CX3:1](=[OX1:2])[OX2H1:3].[OX2H1:4][C:5]>>[CX3:1](=[OX1:2])[OX2:4][C:5].[OX2H2:3]"},
    {"id": "T_AMIDATION", "name": "醯胺化反應 (Amidation)", "smarts": "[CX3:1](=[OX1:2])[OX2H1:3].[NX3H2:4][C,H:5]>>[CX3:1](=[OX1:2])[NX3H1:4][C,H:5].[OX2H2:3]"},
    {"id": "T_SUZUKI", "name": "Suzuki Coupling", "smarts": "[c:1]-[Br,I,Cl].[c:2]-[B](O)O>>[c:1]-[c:2]"},
    {"id": "T_GRIGNARD", "name": "Grignard Addition", "smarts": "[CX3:1](=[OX1:2]).[CX4:3]-[Mg]-[Br,Cl,I]>>[CX4:1](-[OX2H1:2])-[CX4:3]"},
    {"id": "T_ETHERIFICATION", "name": "醚化反應 (Etherification)", "smarts": "[C:1][OX2H1:2].[C:3][OX2H1:4]>>[C:1][OX2:2][C:3]"},
    {"id": "T_DIELS_ALDER", "name": "Diels-Alder 環加成 (Cycloaddition)", "smarts": "[C:1]=[C:2]-[C:3]=[C:4].[C:5]=[C:6]>>[C:1]1-[C:2]=[C:3]-[C:4]-[C:5]-[C:6]1"},
    {"id": "T_FRIEDEL_CRAFTS", "name": "Friedel-Crafts 烷基化 (Alkylation)", "smarts": "[c:1].[C:2]-[Cl,Br,I]>>[c:1]-[C:2]"},
    {"id": "T_HYDROLYSIS_ESTER", "name": "酯水解反應 (Ester Hydrolysis)", "smarts": "[CX3:1](=[OX1:2])[OX2:3][C:4].[O:5]>>[CX3:1](=[OX1:2])[OH].[C:4][OH]"},
    {"id": "T_SCHIFF_BASE", "name": "亞胺生成 (Imine Formation)", "smarts": "[CX3:1]=[OX1:2].[NX3H2:3]>>[CX3:1]=[NX2:3]"},
    {"id": "T_REDUCTION_CARBONYL", "name": "羰基還原反應 (Carbonyl Reduction)", "smarts": "[CX3:1](=[OX1:2])>>[CX4:1]-[OX2H1:2]"},
    {"id": "T_OXIDATION_ALCOHOL", "name": "醇氧化反應 (Alcohol Oxidation)", "smarts": "[CX4H2:1]-[OX2H1:2]>>[CX3:1]=[OX1:2]"}
]

class LocalReactionPredictor:
    def __init__(self):
        self.model = None
        self.templates_dict = {}
        self.is_ml_ready = False
        
        # 預編譯模板
        for t in HIDDEN_TEMPLATES:
            try:
                t['rxn'] = rdChemReactions.ReactionFromSmarts(t['smarts'])
                self.templates_dict[t['id']] = t
            except Exception as e:
                print(f"Error parsing SMARTS for {t['name']}: {e}")
                
        self._train_ml_model()

    def _standardize_smiles(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = Chem.RemoveHs(mol)
                return Chem.MolToSmiles(mol)
        except:
            pass
        return smiles

    def _train_ml_model(self):
        """
        核心 ML 引擎：
        讀取使用者的反應範例，自動找出對應的模板，並訓練隨機森林模型。
        """
        if not RandomForestClassifier:
            print("Warning: scikit-learn is not installed. ML engine disabled.")
            return

        dataset_path = 'reactions_dataset.json'
        if not os.path.exists(dataset_path):
            print("No reactions_dataset.json found. Please provide examples to train the ML engine.")
            return
            
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        X_train = []
        y_train = []
        
        print("Training ML Engine (Auto-Template Mapping)...")
        for data in dataset:
            reactants = data['reactants']
            target_products = set([self._standardize_smiles(s) for s in data['products']])
            
            # 1. 取得反應物的合併特徵 (指紋)
            fp = self._get_combined_fingerprint(reactants)
            if fp is None: continue
            
            mols = [Chem.MolFromSmiles(s) for s in reactants]
            mols = [m for m in mols if m is not None]
            if len(mols) != len(reactants): continue
            
            # 2. 自動尋找能產生目標產物的模板 (Auto-Mapping)
            matched_template_id = None
            for t_id, t in self.templates_dict.items():
                rxn = t.get('rxn')
                if not rxn: continue
                
                # 測試正反向組合
                prods = []
                try:
                    if len(mols) == 2:
                        prods = rxn.RunReactants((mols[0], mols[1]))
                        if not prods: prods = rxn.RunReactants((mols[1], mols[0]))
                    elif len(mols) == 1:
                        prods = rxn.RunReactants((mols[0],))
                except:
                    pass
                
                if prods:
                    # 比較產物是否符合使用者的預期
                    for prod_tuple in prods:
                        prod_smiles = set([self._standardize_smiles(Chem.MolToSmiles(p)) for p in prod_tuple])
                        # 簡單集合交集比較 (只要有產出預期產物即算成功配對)
                        if target_products.intersection(prod_smiles) or prod_smiles == target_products:
                            matched_template_id = t_id
                            break
                if matched_template_id:
                    break
            
            if matched_template_id:
                X_train.append(fp)
                y_train.append(matched_template_id)
        
        if X_train and len(set(y_train)) > 1:
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model.fit(X_train, y_train)
            self.is_ml_ready = True
            print(f"ML Engine Trained Successfully! ({len(X_train)} samples mapped to {len(set(y_train))} templates)")
        elif X_train and len(set(y_train)) == 1:
            # 只有一種反應時的簡單 Fallback
            self.model = y_train[0] # Just store the single ID
            self.is_ml_ready = True
            print("ML Engine Trained (Single template mode)")
        else:
            print("ML Engine failed to map dataset to known templates.")

    def _get_combined_fingerprint(self, smiles_list):
        """將多個反應物合併為單一特徵向量"""
        combined_fp = np.zeros((1024,), dtype=int)
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                # 使用 XOR 或加法結合特徵
                combined_fp = np.bitwise_or(combined_fp, arr.astype(int))
        return combined_fp if np.sum(combined_fp) > 0 else None

    def predict(self, reactant_smiles_list):
        """
        機器學習核心：
        透過 Scikit-Learn 訓練的模型，將新的反應物特徵分類到最適合的歷史反應模板，
        免除了系統必須暴力搜尋數萬條規則的效能瓶頸，也免除了人工撰寫規則的麻煩。
        """
        if not self.is_ml_ready:
            return {"success": False, "message": "ML 模型尚未訓練，請確認是否已提供 reactions_dataset.json 且包含有效的反應範例。"}
            
        fp = self._get_combined_fingerprint(reactant_smiles_list)
        if fp is None:
            return {"success": False, "message": "無法解析反應物 SMILES 產生特徵"}
            
        # 1. 預測最有可能的模板 ID
        predicted_template_id = None
        if isinstance(self.model, RandomForestClassifier):
            predicted_template_id = self.model.predict([fp])[0]
        else:
            predicted_template_id = self.model
            
        t = self.templates_dict.get(predicted_template_id)
        if not t or not t.get('rxn'):
            return {"success": False, "message": "ML 預測了未知的模板。"}
            
        # 2. 執行預測出的反應模板 (A+B 架構的執行層)
        mols = [Chem.MolFromSmiles(s) for s in reactant_smiles_list if s]
        products = []
        try:
            if len(mols) == 2:
                products = t['rxn'].RunReactants((mols[0], mols[1]))
                if not products: products = t['rxn'].RunReactants((mols[1], mols[0]))
            elif len(mols) == 1:
                products = t['rxn'].RunReactants((mols[0],))
        except:
            pass

        if products:
            best_product_smiles = [Chem.MolToSmiles(p) for p in products[0]]
            best_product_smiles = [self._standardize_smiles(s) for s in best_product_smiles if s]
            
            return {
                "success": True, 
                "products": list(set(best_product_smiles)), 
                "template_used": t['name']
            }
        
        return {"success": False, "message": f"ML 模型選擇了 {t['name']}，但化學結構無法成功執行反應。"}
