import numpy as np
import random
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

BOUNDS = {
    "dilation_radius": (2, 20),
    "min_object_size_px": (10, 200),
    "nearby_margin_px": (3, 30),
    "min_skan_branch_length_px": (3, 25),
}

def scale_params(params_dict):
    scaled = []
    for name in ["dilation_radius", "min_object_size_px", "nearby_margin_px", "min_skan_branch_length_px"]:
        low, high = BOUNDS[name]
        val = params_dict[name]
        scaled.append((val - low) / (high - low))
    return np.array(scaled)

def unscale_params(scaled_arr):
    params_dict = {}
    for i, name in enumerate(["dilation_radius", "min_object_size_px", "nearby_margin_px", "min_skan_branch_length_px"]):
        low, high = BOUNDS[name]
        val = scaled_arr[i] * (high - low) + low
        params_dict[name] = int(round(val))
    return params_dict

class TuningSession:
    def __init__(self, session_id: str, frames: list, raw_masks: list):
        self.session_id = session_id
        self.frames = frames  # list of RGB numpy arrays (usually 1 frame to keep overlays simple)
        self.raw_masks = raw_masks  # list of binary masks from CellSAM / zero-shot
        self.X_train = []  # List of scaled arrays (shape: (4,))
        self.y_train = []  # List of float pseudo-scores
        self.candidates = []  # List of 4 current parameter dictionaries
        self.round = 1
        self.max_rounds = 6
        self.created_at = time.time()
        self.history = []  # List of rounds history for debugging or plotting

    def generate_initial_candidates(self):
        cands = []
        # Candidate 0: Baseline defaults
        cands.append({
            "dilation_radius": 8,
            "min_object_size_px": 40,
            "nearby_margin_px": 10,
            "min_skan_branch_length_px": 8
        })

        # Generate 3 diverse candidates
        attempts = 0
        while len(cands) < 4 and attempts < 1000:
            cand = {
                "dilation_radius": random.randint(BOUNDS["dilation_radius"][0], BOUNDS["dilation_radius"][1]),
                "min_object_size_px": random.randint(BOUNDS["min_object_size_px"][0], BOUNDS["min_object_size_px"][1]),
                "nearby_margin_px": random.randint(BOUNDS["nearby_margin_px"][0], BOUNDS["nearby_margin_px"][1]),
                "min_skan_branch_length_px": random.randint(BOUNDS["min_skan_branch_length_px"][0], BOUNDS["min_skan_branch_length_px"][1]),
            }
            scaled_cand = scale_params(cand)
            too_close = False
            for existing in cands:
                scaled_exist = scale_params(existing)
                if np.linalg.norm(scaled_cand - scaled_exist) < 0.35:
                    too_close = True
                    break
            if not too_close:
                cands.append(cand)
            attempts += 1

        # Fallback in case spacing is too tight
        while len(cands) < 4:
            cands.append({
                "dilation_radius": random.randint(BOUNDS["dilation_radius"][0], BOUNDS["dilation_radius"][1]),
                "min_object_size_px": random.randint(BOUNDS["min_object_size_px"][0], BOUNDS["min_object_size_px"][1]),
                "nearby_margin_px": random.randint(BOUNDS["nearby_margin_px"][0], BOUNDS["nearby_margin_px"][1]),
                "min_skan_branch_length_px": random.randint(BOUNDS["min_skan_branch_length_px"][0], BOUNDS["min_skan_branch_length_px"][1]),
            })

        self.candidates = cands
        return cands

    def record_feedback(self, winner_idx: int):
        if not (0 <= winner_idx < len(self.candidates)):
            raise ValueError("Invalid winner index.")

        # Update training data: winner gets 1.0, others get 0.0
        for i, cand in enumerate(self.candidates):
            self.X_train.append(scale_params(cand))
            self.y_train.append(1.0 if i == winner_idx else 0.0)

        self.history.append({
            "round": self.round,
            "candidates": self.candidates.copy(),
            "winner_idx": winner_idx
        })

    def propose_next_candidates(self):
        if len(self.X_train) == 0:
            return self.generate_initial_candidates()

        X = np.array(self.X_train)
        y = np.array(self.y_train)

        # Fit GP Regressor
        kernel = Matern(length_scale=[0.25, 0.25, 0.25, 0.25], nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)

        # Generate large random pool of candidates
        pool_size = 1000
        pool_dicts = []
        pool_scaled = []
        for _ in range(pool_size):
            cand = {
                "dilation_radius": random.randint(BOUNDS["dilation_radius"][0], BOUNDS["dilation_radius"][1]),
                "min_object_size_px": random.randint(BOUNDS["min_object_size_px"][0], BOUNDS["min_object_size_px"][1]),
                "nearby_margin_px": random.randint(BOUNDS["nearby_margin_px"][0], BOUNDS["nearby_margin_px"][1]),
                "min_skan_branch_length_px": random.randint(BOUNDS["min_skan_branch_length_px"][0], BOUNDS["min_skan_branch_length_px"][1]),
            }
            pool_dicts.append(cand)
            pool_scaled.append(scale_params(cand))

        X_pool = np.array(pool_scaled)
        mu, sigma = gp.predict(X_pool, return_std=True)

        # Calculate UCB: explore parameters with high uncertainty or high predicted success
        ucb = mu + 2.0 * sigma

        # Select 4 diverse candidates using greedy farthest selection on UCB
        selected_indices = []
        selected_scaled = []

        # Sort indices by UCB descending
        sorted_idx = np.argsort(ucb)[::-1]

        for idx in sorted_idx:
            cand_scaled = X_pool[idx]
            too_close = False
            for sel in selected_scaled:
                if np.linalg.norm(cand_scaled - sel) < 0.25:
                    too_close = True
                    break
            
            # Avoid training data points
            for tr in X:
                if np.linalg.norm(cand_scaled - tr) < 0.05:
                    too_close = True
                    break

            if not too_close:
                selected_indices.append(idx)
                selected_scaled.append(cand_scaled)
                if len(selected_indices) == 4:
                    break

        # Fallback if we couldn't find 4 diverse candidates
        if len(selected_indices) < 4:
            for idx in sorted_idx:
                if idx not in selected_indices:
                    selected_indices.append(idx)
                    if len(selected_indices) == 4:
                        break

        self.candidates = [pool_dicts[idx] for idx in selected_indices]
        self.round += 1
        return self.candidates

    def get_best_recommendation(self):
        if len(self.X_train) == 0:
            return self.candidates[0], 0.0

        X = np.array(self.X_train)
        y = np.array(self.y_train)

        kernel = Matern(length_scale=[0.25, 0.25, 0.25, 0.25], nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=5, random_state=42)
        gp.fit(X, y)

        # Generate large random pool of candidates to evaluate
        pool_size = 1000
        pool_dicts = []
        pool_scaled = []
        for _ in range(pool_size):
            cand = {
                "dilation_radius": random.randint(BOUNDS["dilation_radius"][0], BOUNDS["dilation_radius"][1]),
                "min_object_size_px": random.randint(BOUNDS["min_object_size_px"][0], BOUNDS["min_object_size_px"][1]),
                "nearby_margin_px": random.randint(BOUNDS["nearby_margin_px"][0], BOUNDS["nearby_margin_px"][1]),
                "min_skan_branch_length_px": random.randint(BOUNDS["min_skan_branch_length_px"][0], BOUNDS["min_skan_branch_length_px"][1]),
            }
            pool_dicts.append(cand)
            pool_scaled.append(scale_params(cand))

        X_pool = np.array(pool_scaled)
        mu = gp.predict(X_pool)
        
        best_idx = np.argmax(mu)
        return pool_dicts[best_idx], float(mu[best_idx])
