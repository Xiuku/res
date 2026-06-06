import json
from local_engine import LocalReactionPredictor, HIDDEN_TEMPLATES
# pyrefly: ignore [missing-import]
from rdkit import Chem
# pyrefly: ignore [missing-import]
from rdkit.Chem import rdChemReactions

with open('reactions_dataset.json', 'r') as f:
    dataset = json.load(f)

for data in dataset:
    reactants = data['reactants']
    target_products = set(data['products'])
    print(f"\nReactants: {reactants}, Target: {target_products}")
    
    mols = [Chem.MolFromSmiles(s) for s in reactants]
    
    for t in HIDDEN_TEMPLATES:
        rxn = rdChemReactions.ReactionFromSmarts(t['smarts'])
        try:
            prods = rxn.RunReactants(tuple(mols))
            if not prods: prods = rxn.RunReactants(tuple(reversed(mols)))
            if prods:
                prod_smiles = set([Chem.MolToSmiles(p) for p in prods[0]])
                print(f"  Template {t['name']} produced: {prod_smiles}")
        except Exception as e:
            pass
