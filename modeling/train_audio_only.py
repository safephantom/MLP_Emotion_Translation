import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import KEMDy19MultiModalDataset, multimodal_collate_fn
from model import KEMDy19MultiModalModel
from tqdm import tqdm

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths (adjust as needed)
    dataset_csv = 'merged_dataset_soft_fixed.csv'
    ef_emotion_csv = 'dynamic_ef_weights_fixed.csv'
    audio_dir = '../cached_features'
    
    print("Loading datasets...")
    train_dataset = KEMDy19MultiModalDataset(dataset_csv, ef_emotion_csv, audio_dir, split='train')
    val_dataset = KEMDy19MultiModalDataset(dataset_csv, ef_emotion_csv, audio_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=multimodal_collate_fn, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=multimodal_collate_fn, num_workers=2, pin_memory=True)
    
    # Optuna에서 찾은 최적의 은닉층 크기와 드롭아웃 적용
    model = KEMDy19MultiModalModel(audio_hidden_dim=128).to(device)
    model.audio_lstm.dropout = 0.4189797615156061
    model.emotion_head[2].p = 0.4189797615156061
    
    # Loss functions
    criterion_emotion = nn.CrossEntropyLoss()
    criterion_regression = nn.MSELoss()
    
    # Optuna에서 찾은 최적의 학습률(lr)과 가중치 감소(weight_decay) 적용
    optimizer = optim.Adam(model.parameters(), lr=0.003955303736003501, weight_decay=0.00026042163602817516)
    
    # 에포크를 30으로 늘려서 리즈 시절(가장 점수가 높은 순간)의 가중치를 저장하도록 유도
    num_epochs = 30
    best_val_loss = float('inf')
    
    # Weights for multi-task loss
    w_e, w_v, w_a = 1.0, 1.0, 1.0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            (audios, lengths, efs), (e_labels, v_labels, a_labels) = batch
            audios, lengths, efs = audios.to(device), lengths.to(device), efs.to(device)
            e_labels, v_labels, a_labels = e_labels.to(device), v_labels.to(device), a_labels.to(device)
            
            # Ablation Study: 텍스트(EF) 정보를 완벽하게 차단 (모두 1/7의 균등 분포로 덮어쓰기)
            efs = torch.full_like(efs, 1.0/7.0).to(device)
            
            optimizer.zero_grad()
            
            e_pred, v_pred, a_pred = model(audios, lengths, efs)
            
            loss_e = criterion_emotion(e_pred, e_labels)
            loss_v = criterion_regression(v_pred, v_labels)
            loss_a = criterion_regression(a_pred, a_labels)
            
            total_loss = w_e * loss_e + w_v * loss_v + w_a * loss_a
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_loss_e, val_loss_v, val_loss_a = 0.0, 0.0, 0.0
        
        correct_e = 0
        total_soft_acc = 0.0
        total_e = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                (audios, lengths, efs), (e_labels, v_labels, a_labels) = batch
                audios, lengths, efs = audios.to(device), lengths.to(device), efs.to(device)
                e_labels, v_labels, a_labels = e_labels.to(device), v_labels.to(device), a_labels.to(device)
                
                # Ablation Study: 검증할 때도 텍스트(EF) 정보 완벽 차단
                efs = torch.full_like(efs, 1.0/7.0).to(device)
                
                e_pred, v_pred, a_pred = model(audios, lengths, efs)
                
                loss_e = criterion_emotion(e_pred, e_labels)
                loss_v = criterion_regression(v_pred, v_labels)
                loss_a = criterion_regression(a_pred, a_labels)
                
                total_loss = w_e * loss_e + w_v * loss_v + w_a * loss_a
                val_loss += total_loss.item()
                
                val_loss_e += loss_e.item()
                val_loss_v += loss_v.item()
                val_loss_a += loss_a.item()
                
                # Accuracy for emotion
                _, predicted = torch.max(e_pred.data, 1)
                _, true_e = torch.max(e_labels, 1)
                total_e += e_labels.size(0)
                correct_e += (predicted == true_e).sum().item()
                
                # Soft Accuracy (Histogram Intersection between predicted probs and true probs)
                e_pred_probs = torch.softmax(e_pred.data, dim=1)
                intersection = torch.min(e_pred_probs, e_labels).sum(dim=1)
                total_soft_acc += intersection.sum().item()
                
        val_loss /= len(val_loader)
        val_acc = correct_e / total_e * 100
        val_soft_acc = total_soft_acc / total_e * 100
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} (E:{val_loss_e/len(val_loader):.4f}, V:{val_loss_v/len(val_loader):.4f}, A:{val_loss_a/len(val_loader):.4f})")
        print(f"  --> Hard Acc: {val_acc:.2f}%, Soft Acc: {val_soft_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'kemdy19_audio_only.pth')
            print("  --> Saved best AUDIO-ONLY model!")
            
    print("Training finished.")

if __name__ == "__main__":
    train()
