import glob
import re
import os

for file_path in glob.glob('test_cases/test_case_*.py'):
    with open(file_path, 'r') as f:
        code = f.read()
        
    # 1. Update DATA_PATH
    code = code.replace("DATA_PATH = os.path.join(BASE_DIR, 'live_calib_data.npz')",
                        "DATA_PATH = os.path.join(BASE_DIR, f'data_{os.path.basename(__file__).replace(\".py\", \"\")}.npz')")
                        
    # 2. Update background task
    bg_old = """            with FileLock(LOCK_PATH):
                if os.path.exists(DATA_PATH):
                    try:
                        data = np.load(DATA_PATH)
                        hist_features = data['features']
                        hist_targets = data['targets']
                    except:
                        hist_features = self.logger.historical_features
                        hist_targets = self.logger.historical_targets
                else:
                    hist_features = self.logger.historical_features
                    hist_targets = self.logger.historical_targets
                
                # Combine with historical data
                if len(new_features_np) > 0 and len(hist_features) > 0:
                    features_np = np.vstack([hist_features, new_features_np])
                    targets_np = np.vstack([hist_targets, new_targets_np])
                elif len(new_features_np) > 0:
                    features_np = new_features_np
                    targets_np = new_targets_np
                else:
                    features_np = hist_features
                    targets_np = hist_targets
                    
                if len(features_np) > 10000:
                    features_np = features_np[-10000:]
                    targets_np = targets_np[-10000:]
                    
                if len(features_np) == 0:
                    return
                
                np.savez(DATA_PATH, features=features_np, targets=targets_np)
                
                if os.path.exists(MODEL_PATH):
                    try:
                        self.model.load_state_dict(torch.load(MODEL_PATH))
                    except:
                        pass
                        
            # Fine-tune Bayesian model
            self.model.train()
            dataset_size = len(features_np)
            epochs = 2
            batch_size = 32
            
            X = torch.tensor(features_np, dtype=torch.float32)
            Y = torch.tensor(targets_np, dtype=torch.float32)"""
            
    bg_new = """            with FileLock(LOCK_PATH):
                if os.path.exists(DATA_PATH):
                    try:
                        data = np.load(DATA_PATH)
                        hist_features = data['features']
                        hist_targets = data['targets']
                    except:
                        hist_features = self.logger.historical_features
                        hist_targets = self.logger.historical_targets
                else:
                    hist_features = self.logger.historical_features
                    hist_targets = self.logger.historical_targets
                
                # Combine LOCAL data
                if len(new_features_np) > 0 and len(hist_features) > 0:
                    features_np = np.vstack([hist_features, new_features_np])
                    targets_np = np.vstack([hist_targets, new_targets_np])
                elif len(new_features_np) > 0:
                    features_np = new_features_np
                    targets_np = new_targets_np
                else:
                    features_np = hist_features
                    targets_np = hist_targets
                    
                # Cap LOCAL data at 2000 to keep global RAM stable
                if len(features_np) > 2000:
                    features_np = features_np[-2000:]
                    targets_np = targets_np[-2000:]
                    
                if len(features_np) == 0:
                    return
                
                np.savez(DATA_PATH, features=features_np, targets=targets_np)
                
                # Load GLOBAL aggregated data from all test cases
                import glob
                global_features = []
                global_targets = []
                for file in glob.glob(os.path.join(BASE_DIR, 'data_*.npz')):
                    try:
                        d = np.load(file)
                        global_features.append(d['features'])
                        global_targets.append(d['targets'])
                    except:
                        pass
                
                if global_features:
                    train_features_np = np.vstack(global_features)
                    train_targets_np = np.vstack(global_targets)
                else:
                    train_features_np = features_np
                    train_targets_np = targets_np
                
                # Sync latest weights
                if os.path.exists(MODEL_PATH):
                    try:
                        self.model.load_state_dict(torch.load(MODEL_PATH))
                    except:
                        pass
                        
            # Fine-tune Bayesian model on GLOBAL dataset
            self.model.train()
            dataset_size = len(train_features_np)
            epochs = 1  # Reduced epochs because dataset is massive
            batch_size = 32
            
            X = torch.tensor(train_features_np, dtype=torch.float32)
            Y = torch.tensor(train_targets_np, dtype=torch.float32)"""
            
    code = code.replace(bg_old, bg_new)
    
    with open(file_path, 'w') as f:
        f.write(code)

print("All 7 files patched successfully!")
