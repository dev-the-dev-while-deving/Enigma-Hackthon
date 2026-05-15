import re

def patch_file(file_path):
    with open(file_path, 'r') as f:
        code = f.read()

    # Add FileLock import and LOCK_PATH
    if 'from filelock import FileLock' not in code:
        code = code.replace('from captum.attr import IntegratedGradients', 
                            'from captum.attr import IntegratedGradients\nfrom filelock import FileLock')
        code = code.replace("DATA_PATH = os.path.join(BASE_DIR, 'live_calib_data.npz')",
                            "DATA_PATH = os.path.join(BASE_DIR, 'live_calib_data.npz')\nLOCK_PATH = os.path.join(BASE_DIR, 'training.lock')")
        
        # Also fix oncoming paths back to main paths
        code = code.replace("MODEL_LOAD_PATH = os.path.join(BASE_DIR, 'bayesian_model.pt')", "MODEL_PATH = os.path.join(BASE_DIR, 'bayesian_model.pt')")
        code = code.replace("MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'bayesian_model_oncoming.pt')", "")
        code = code.replace("DATA_PATH = os.path.join(BASE_DIR, 'live_calib_data_oncoming.npz')", "DATA_PATH = os.path.join(BASE_DIR, 'live_calib_data.npz')\nLOCK_PATH = os.path.join(BASE_DIR, 'training.lock')")
        
        code = code.replace("MODEL_LOAD_PATH", "MODEL_PATH")
        code = code.replace("MODEL_SAVE_PATH", "MODEL_PATH")

    # Update Data Logger to use lock
    logger_init_old = """        # Load historical data if it exists so we don't overwrite it across restarts
        if os.path.exists(DATA_PATH):
            try:
                data = np.load(DATA_PATH)"""
    
    logger_init_new = """        # Load historical data if it exists so we don't overwrite it across restarts
        with FileLock(LOCK_PATH):
            if os.path.exists(DATA_PATH):
                try:
                    data = np.load(DATA_PATH)"""
    code = code.replace(logger_init_old, logger_init_new)

    # Update _background_task
    bg_task_old = """            # Combine with historical data
            if len(new_features_np) > 0 and len(self.logger.historical_features) > 0:
                features_np = np.vstack([self.logger.historical_features, new_features_np])
                targets_np = np.vstack([self.logger.historical_targets, new_targets_np])
            elif len(new_features_np) > 0:
                features_np = new_features_np
                targets_np = new_targets_np
            else:
                features_np = self.logger.historical_features
                targets_np = self.logger.historical_targets
                
            if len(features_np) == 0:
                return
            
            # Save the combined continuous live_calib_data.npz
            np.savez(DATA_PATH, features=features_np, targets=targets_np)
            
            # Fine-tune Bayesian model
            self.model.train()"""
            
    bg_task_new = """            with FileLock(LOCK_PATH):
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
            self.model.train()"""
    
    code = code.replace(bg_task_old, bg_task_new)
    
    # Update torch save
    torch_save_old = """            # Save model checkpoint
            torch.save(self.model.state_dict(), MODEL_PATH)
            print(f"\\n✅ Processing Layer updated"""
            
    torch_save_new = """            # Save model checkpoint
            with FileLock(LOCK_PATH):
                torch.save(self.model.state_dict(), MODEL_PATH)
            print(f"\\n✅ Processing Layer updated"""
            
    code = code.replace(torch_save_old, torch_save_new)
    
    # Also fix oncoming isolated save comment
    torch_save_oncoming_old = """            # Save model checkpoint (isolated to this test case)
            torch.save(self.model.state_dict(), MODEL_PATH)
            print(f"\\n✅ Processing Layer updated"""
            
    code = code.replace(torch_save_oncoming_old, torch_save_new)

    with open(file_path, 'w') as f:
        f.write(code)

patch_file('test_cases/test_case_mpc.py')
patch_file('test_cases/test_case_oncoming.py')
