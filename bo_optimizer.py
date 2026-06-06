import os
import pandas as pd
import numpy as np

try:
    # 嘗試匯入官方 edbo 套件
    # pyrefly: ignore [missing-import]
    from edbo.bro import BO
    EDBO_AVAILABLE = True
except ImportError as e:
    EDBO_AVAILABLE = False
    print(f"Warning: edbo is not installed or failed to load. Bayesian Optimization engine disabled. Error: {e}")

class LocalBayesianOptimizer:
    """
    與 local_engine.py 中 LocalReactionPredictor 相同的封裝架構。
    專門用來處理化學反應條件的最佳化 (Experimental Design via Bayesian Optimization)。
    """
    def __init__(self):
        self.is_bo_ready = EDBO_AVAILABLE

    def suggest_next_experiment(self, parameters_space, historical_results, target_col="yield", maximize=True):
        """
        利用 EDBO 建議下一個最優的實驗條件。
        
        :param parameters_space: dict, 定義各個變數的搜尋空間。
                                 例如: {'temperature': [20, 25, 30, ...], 'concentration': [0.1, 0.2, ...]}
        :param historical_results: list of dict, 過去的實驗結果。
                                 例如: [{'temperature': 25, 'concentration': 0.1, 'yield': 45.0}, ...]
        :param target_col: str, 欲最佳化的目標欄位名稱。
        :param maximize: bool, 是否為最大化目標。
        """
        if not historical_results:
            return {"success": False, "message": "請提供至少一筆以上的歷史實驗數據。"}

        # 如果 EDBO 不可用，使用內建的備用啟發式優化算法
        if not self.is_bo_ready:
            fallback_proposals = self._fallback_suggest(parameters_space, historical_results, target_col, maximize)
            if fallback_proposals:
                return {
                    "success": True,
                    "next_experiment": fallback_proposals[0],
                    "all_proposals": fallback_proposals,
                    "is_fallback": True,
                    "message": "成功利用備用啟發式優化器推導出推薦實驗條件（本機未安裝或載入 EDBO 套件）。"
                }
            else:
                return {"success": False, "message": "備用優化器未能生成建議條件。"}
            
        try:
            # 將輸入資料轉換為 pandas DataFrame
            results_df = pd.DataFrame(historical_results)
            
            # parameters_space 定義了完整的反應空間 (Reaction Space)
            space_df = self._build_reaction_space(parameters_space)
            
            # 初始化 EDBO 的 BO 物件
            bo = BO(
                results=results_df, 
                domain=space_df, 
                target=target_col
            )
            
            # 執行高斯過程 (Gaussian Process) 模型訓練與採集函數計算
            bo.run()
            
            # 取得建議的下一步實驗條件
            proposals = bo.proposed_experiments
            proposals_list = proposals.to_dict(orient='records')
            
            if not proposals_list:
                raise ValueError("EDBO 沒有回傳任何候選實驗條件。")
                
            next_experiment = proposals_list[0]
            
            return {
                "success": True,
                "next_experiment": next_experiment,
                "all_proposals": proposals_list,
                "is_fallback": False,
                "message": "成功利用 EDBO 貝氏優化推導出最佳的下一步實驗條件！"
            }
            
        except Exception as e:
            # 若 EDBO 優化失敗，自動切換至備用優化演算法以提高強健性
            print(f"EDBO 優化失敗，切換至備用算法。錯誤: {str(e)}")
            fallback_proposals = self._fallback_suggest(parameters_space, historical_results, target_col, maximize)
            if fallback_proposals:
                return {
                    "success": True,
                    "next_experiment": fallback_proposals[0],
                    "all_proposals": fallback_proposals,
                    "is_fallback": True,
                    "message": f"EDBO 計算失敗，已切換至備用優化器生成推薦條件。原因: {str(e)}"
                }
            return {
                "success": False,
                "message": f"EDBO 與備用優化器均發生錯誤: {str(e)}"
            }

    def _fallback_suggest(self, parameters_space, historical_results, target_col, maximize):
        """
        備用優化演算法：
        在 EDBO 套件不可用或出錯時，透過簡單的 Exploitation-Exploration (開發與探索) 啟發式評分，
        推薦下一步實驗條件。
        """
        import itertools
        
        # 1. 產生所有候選組合
        keys = list(parameters_space.keys())
        values = list(parameters_space.values())
        combinations = list(itertools.product(*values))
        
        # 轉成 dict 清單方便處理
        candidates = [dict(zip(keys, comb)) for comb in combinations]
        
        # 2. 找出已測試的歷史條件
        tested_sets = []
        for res in historical_results:
            tested_sets.append({k: res.get(k) for k in keys if k in res})
            
        # 過濾出未測試的候選條件
        untested_candidates = []
        for cand in candidates:
            is_tested = False
            for tested in tested_sets:
                match = True
                for k in keys:
                    v_cand = cand[k]
                    v_test = tested.get(k)
                    if isinstance(v_cand, (int, float)) and isinstance(v_test, (int, float)):
                        if not np.isclose(v_cand, v_test, atol=1e-5):
                            match = False
                            break
                    else:
                        if v_cand != v_test:
                            match = False
                            break
                if match:
                    is_tested = True
                    break
            if not is_tested:
                untested_candidates.append(cand)
                
        # 如果所有可能都測試過了，重新使用全空間
        if not untested_candidates:
            untested_candidates = candidates
            
        # 3. 如果歷史結果不足，直接隨機挑選幾個
        if not historical_results:
            np.random.shuffle(untested_candidates)
            return untested_candidates[:min(5, len(untested_candidates))]
            
        # 4. 尋找最佳歷史點
        try:
            valid_results = [r for r in historical_results if target_col in r and r[target_col] is not None]
            if not valid_results:
                np.random.shuffle(untested_candidates)
                return untested_candidates[:min(5, len(untested_candidates))]
                
            # 排序找出最優點
            sorted_results = sorted(valid_results, key=lambda x: float(x[target_col]), reverse=maximize)
            best_point = sorted_results[0]
            
            # 5. 對每個未測試的候選點評分 (Exploration vs Exploitation)
            scored_candidates = []
            for cand in untested_candidates:
                dist_to_best = 0.0
                min_dist_to_history = float('inf')
                
                # 計算與最佳點的歸一化歐氏距離
                for k in keys:
                    vals = parameters_space[k]
                    numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                    span = (max(numeric_vals) - min(numeric_vals)) if len(numeric_vals) > 1 else 1.0
                    if span == 0: span = 1.0
                    
                    v_cand = cand[k]
                    v_best = best_point.get(k)
                    
                    if isinstance(v_cand, (int, float)) and isinstance(v_best, (int, float)):
                        d_best = abs(v_cand - v_best) / span
                    else:
                        d_best = 0.0 if v_cand == v_best else 1.0
                    dist_to_best += d_best ** 2
                
                dist_to_best = np.sqrt(dist_to_best)
                
                # 計算到所有已測試歷史點的最小歐氏距離 (Exploration)
                for hist in valid_results:
                    hist_dist = 0.0
                    for k in keys:
                        vals = parameters_space[k]
                        numeric_vals = [v for v in vals if isinstance(v, (int, float))]
                        span = (max(numeric_vals) - min(numeric_vals)) if len(numeric_vals) > 1 else 1.0
                        if span == 0: span = 1.0
                        
                        v_cand = cand[k]
                        v_hist = hist.get(k)
                        if isinstance(v_cand, (int, float)) and isinstance(v_hist, (int, float)):
                            d = (v_cand - v_hist) / span
                        else:
                            d = 0.0 if v_cand == v_hist else 1.0
                        hist_dist += d ** 2
                    hist_dist = np.sqrt(hist_dist)
                    if hist_dist < min_dist_to_history:
                        min_dist_to_history = hist_dist
                
                # 評分公式：-dist_to_best (越接近最好點，分數越高) + 0.3 * min_dist_to_history (越遠離已知點，越有探索價值)
                score = -dist_to_best + 0.3 * min_dist_to_history
                scored_candidates.append((cand, score))
                
            # 排序候選點
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return [x[0] for x in scored_candidates[:min(5, len(scored_candidates))]]
            
        except Exception as e:
            print(f"備用優化器評分出錯: {e}")
            np.random.shuffle(untested_candidates)
            return untested_candidates[:min(5, len(untested_candidates))]

    def _build_reaction_space(self, param_dict):
        """
        將各維度的參數展開為完整的實驗空間 DataFrame (笛卡爾積)
        """
        import itertools
        keys = list(param_dict.keys())
        values = list(param_dict.values())
        
        # 產生所有組合
        combinations = list(itertools.product(*values))
        
        # 建立 DataFrame
        space_df = pd.DataFrame(combinations, columns=keys)
        return space_df

