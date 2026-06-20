import time
import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import KEMDy19MultiModalDataset, multimodal_collate_fn
from model import KEMDy19MultiModalModel

def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    audio_hidden_dim = trial.suggest_categorical("audio_hidden_dim", [64, 128, 256])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    dataset_csv = 'merged_dataset_soft_fixed.csv'
    ef_emotion_csv = 'dynamic_ef_weights_fixed.csv'
    audio_dir = '../cached_features'
    
    train_dataset = KEMDy19MultiModalDataset(dataset_csv, ef_emotion_csv, audio_dir, split='train')
    val_dataset = KEMDy19MultiModalDataset(dataset_csv, ef_emotion_csv, audio_dir, split='val')
    
    # CPU 일꾼 2명(Colab 호환성 및 속도 최적화)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multimodal_collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=multimodal_collate_fn, num_workers=2, pin_memory=True)
    
    model = KEMDy19MultiModalModel(audio_hidden_dim=audio_hidden_dim).to(device)
    model.audio_lstm.dropout = dropout_rate
    model.emotion_head[2].p = dropout_rate
    
    criterion_emotion = nn.CrossEntropyLoss()
    criterion_regression = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    num_epochs = 30 
    best_val_loss = float('inf') 
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time() # 에포크 시작 시간 기록
        
        model.train()
        for batch in train_loader:
            (audios, lengths, efs), (e_labels, v_labels, a_labels) = batch
            audios, lengths, efs = audios.to(device), lengths.to(device), efs.to(device)
            e_labels, v_labels, a_labels = e_labels.to(device), v_labels.to(device), a_labels.to(device)
            
            optimizer.zero_grad()
            e_pred, v_pred, a_pred = model(audios, lengths, efs)
            
            loss = criterion_emotion(e_pred, e_labels) + criterion_regression(v_pred, v_labels) + criterion_regression(a_pred, a_labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                (audios, lengths, efs), (e_labels, v_labels, a_labels) = batch
                audios, lengths, efs = audios.to(device), lengths.to(device), efs.to(device)
                e_labels, v_labels, a_labels = e_labels.to(device), v_labels.to(device), a_labels.to(device)
                
                e_pred, v_pred, a_pred = model(audios, lengths, efs)
                loss = criterion_emotion(e_pred, e_labels) + criterion_regression(v_pred, v_labels) + criterion_regression(a_pred, a_labels)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        elapsed_time = time.time() - epoch_start_time
        print(f"[Trial {trial.number}] Epoch {epoch+1:02d}/{num_epochs:02d} | Val Loss: {val_loss:.4f} | Time: {elapsed_time:.1f}s")
        
        # 이번 에포크의 점수가 지금까지 최고 기록이라면 갱신
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned after {epoch+1} epochs due to unpromising results.")
            raise optuna.exceptions.TrialPruned()
            
    # 무조건 '가장 낮았던 최고 점수'를 반환
    return best_val_loss

if __name__ == "__main__":
    # Optuna Study 생성 (minimize: Val Loss가 작을수록 좋음)
    study = optuna.create_study(direction="minimize")
    
    print("하이퍼파라미터 최적화(Optuna)를 시작합니다...")
    # 밤샘 훈련을 위한 최적의 횟수: 50번
    study.optimize(objective, n_trials=50) 
    
    print("\n==================================")
    print("최적의 하이퍼파라미터 조합을 찾았습니다!")
    trial = study.best_trial
    print(f"가장 낮았던 검증 손실(Val Loss): {trial.value:.4f}")
    print("적용해야 할 하이퍼파라미터 값:")
    for key, value in trial.params.items():
        print(f"  - {key}: {value}")
    print("==================================")
