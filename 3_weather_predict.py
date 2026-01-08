import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

CONFIG = {
    "file_path": "Desktop/DM_2025_Dataset/weather.csv",
    "window_size": 12, "batch_size": 64, "hidden_dim": 128, "epochs": 40,
    "lr": 0.001, "weight_decay": 1e-4, "corr_threshold": 0.3, "train_ratio": 0.8, "loss_alpha": 0.7
}

# 数据加载与预处理
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    encodings = ['gbk', 'utf-8', 'latin-1', 'gb2312']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, parse_dates=["date"], index_col="date", encoding=enc)
            break
        except:
            continue
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how="all").bfill().ffill()
    if df.isnull().any().any():
        df = df.fillna(df.mean())
    return df

# 特征工程
def create_features(df, window_size):
    df = df.copy()
    # 统计特征
    df['OT_2h_mean'] = df['OT'].rolling(window=window_size).mean()
    df['OT_2h_std'] = df['OT'].rolling(window=window_size).std()
    df['OT_2h_min'] = df['OT'].rolling(window=window_size).min()
    df['OT_2h_max'] = df['OT'].rolling(window=window_size).max()
    # 变化特征
    df['OT_diff_1'] = df['OT'].diff(1).fillna(0)
    df['OT_diff_3'] = df['OT'].diff(3).fillna(0)
    df['OT_diff_6'] = df['OT'].diff(6).fillna(0)
    df['OT_diff_12'] = df['OT'].diff(12).fillna(0)
    # 变化率
    df['OT_pct_1'] = df['OT'].pct_change(1).fillna(0)
    df['OT_pct_3'] = df['OT'].pct_change(3).fillna(0)
    df['OT_pct_6'] = df['OT'].pct_change(6).fillna(0)
    # 时间特征
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / (24*60))
    df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / (24*60))
    # 交互特征
    for feat in ['T (degC)', 'Tdew (degC)', 'rh (%)', 'Tlog (degC)']:
        if feat in df.columns:
            df[f'{feat}_OT_ratio'] = df[feat] / (df['OT'] + 1e-6)
            df[f'{feat}_diff'] = df[feat].diff(1).fillna(0)
    df = df.bfill().ffill()
    return df

# 标准化函数
def safe_standard_scaler(data):
    scaler = StandardScaler()
    std = np.std(data, axis=0)
    const_mask = std < 1e-6
    if np.any(const_mask):
        non_const_data = data[:, ~const_mask]
        non_const_scaled = scaler.fit_transform(non_const_data)
        scaled_data = np.zeros_like(data)
        scaled_data[:, ~const_mask] = non_const_scaled
        scaled_data[:, const_mask] = data[:, const_mask]
        return scaler, scaled_data
    return scaler, scaler.fit_transform(data)

# 创建序列数据
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i-window_size:i])
        y_seq.append(y[i, 0])
    return np.array(X_seq), np.array(y_seq)

# LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, ot_feature_idx=None):
        super().__init__()
        self.ot_feature_idx = ot_feature_idx
        # 双向LSTM
        self.bilstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim//2, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=0.2)
        # 注意力机制
        self.temporal_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2), nn.Tanh(), nn.Linear(hidden_dim//2, 1))
        self.feature_attention = nn.Sequential(nn.Linear(input_dim, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))
        # 预测头
        self.residual_head = nn.Sequential(nn.Linear(hidden_dim+1, 64), nn.ReLU(), nn.Dropout(0.2),
                                           nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.direct_head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
        self.gate = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.shape
        # OT特征提取
        current_ot = x[:, -1, self.ot_feature_idx:self.ot_feature_idx+1] if (self.ot_feature_idx is not None and self.ot_feature_idx < feat_dim) else x[:, -1, 0:1]
        # LSTM编码
        lstm_out, _ = self.bilstm(x)
        # 时间注意力
        temporal_weights = torch.softmax(self.temporal_attention(lstm_out), dim=1)
        temporal_context = torch.sum(temporal_weights * lstm_out, dim=1)
        # 特征注意力
        last_step = x[:, -1, :]
        feature_weights = torch.softmax(self.feature_attention(last_step), dim=0)
        feature_context = torch.sum(feature_weights * last_step, dim=1, keepdim=True)
        # 融合预测
        combined_context = torch.cat([temporal_context, feature_context], dim=1)
        residual_pred = self.residual_head(combined_context)
        direct_pred = self.direct_head(temporal_context)
        gate_weight = self.gate(temporal_context)
        final_pred = gate_weight * (current_ot + residual_pred) + (1 - gate_weight) * direct_pred
        return final_pred

# 数据集类
class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 损失函数
def combined_loss(y_pred, y_true, alpha=0.7):
    mse = nn.MSELoss()(y_pred, y_true)
    mae = nn.L1Loss()(y_pred, y_true)
    return alpha * mse + (1 - alpha) * mae

# 主流程
if __name__ == "__main__":
    # 数据加载
    df = load_data(CONFIG["file_path"])
    df = create_features(df, CONFIG["window_size"])
    print(f"数据形状: {df.shape}, 特征数: {len(df.columns)}")

    # 特征选择与标准化
    target_col = 'OT'
    feature_cols = [col for col in df.columns if col != target_col]
    # 相关性筛选
    correlations = {col: df[col].corr(df[target_col]) for col in feature_cols if not np.isinf(df[col].corr(df[target_col]))}
    high_corr_features = [col for col, corr in correlations.items() if abs(corr) > CONFIG["corr_threshold"]]
    # 特征数量保护
    if len(high_corr_features) < 5:
        high_corr_features = [col for col, _ in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]]
    elif len(high_corr_features) > 50:
        high_corr_features = [col for col, _ in sorted([(col, correlations[col]) for col in high_corr_features], key=lambda x: abs(x[1]), reverse=True)[:50]]
    
    # 标准化
    X_features = df[high_corr_features].values
    y_target = df[[target_col]].values
    scaler_X, X_scaled = safe_standard_scaler(X_features)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_target)

    # 序列创建与划分
    X_sequences, y_sequences = create_sequences(X_scaled, y_scaled, CONFIG["window_size"])
    train_size = int(CONFIG["train_ratio"] * len(X_sequences))
    X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
    y_train, y_test = y_sequences[:train_size], y_sequences[train_size:]
    test_original_indices = df.index[train_size + CONFIG["window_size"]:train_size + CONFIG["window_size"] + len(X_test)]
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

    # 模型初始化
    ot_feature_idx = high_corr_features.index([col for col in high_corr_features if col.startswith('OT_')][0]) if [col for col in high_corr_features if col.startswith('OT_')] else 0
    model = LSTM(input_dim=X_train.shape[2], hidden_dim=CONFIG["hidden_dim"], ot_feature_idx=ot_feature_idx).to(device)

    # 数据加载器
    train_loader = DataLoader(WeatherDataset(X_train, y_train), batch_size=CONFIG["batch_size"], shuffle=True, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(WeatherDataset(X_test, y_test), batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=torch.cuda.is_available())

    # 训练配置
    criterion = lambda y_pred, y_true: combined_loss(y_pred, y_true, alpha=CONFIG["loss_alpha"])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=1e-5)

    # 训练循环
    best_val_loss = float('inf')
    print("\n开始训练:")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=True)
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
        
        # 进度更新
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(test_loader.dataset)
        scheduler.step()
        print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        best_val_loss = min(best_val_loss, avg_val_loss)

    # 模型评估
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            y_pred_list.extend(outputs.cpu().numpy())
            y_true_list.extend(batch_y.cpu().numpy())
    
    # 反标准化
    y_pred = scaler_y.inverse_transform(np.array(y_pred_list).reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(np.array(y_true_list).reshape(-1, 1)).flatten()
    
    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan

    # 输出结果
    print(f"{'MAE':<15} {mae:<10.2f}")
    print(f"{'MSE':<15} {mse:<10.2f}")
    print(f"{'RMSE':<15} {rmse:<10.2f}")
    print(f"{'R²':<15} {r2:<10.4f}")
    print(f"{'相关系数':<15} {corr:<10.4f}")
    if not np.isnan(mape):
        print(f"{'MAPE (%)':<15} {mape:<10.2f}")

    results_df = pd.DataFrame({
        'datetime': test_original_indices[:len(y_true)],
        'y_true': y_true, 'y_pred': y_pred,
        'error': y_pred - y_true, 'abs_error': np.abs(y_pred - y_true)
    })
    results_df.to_csv("weather_prediction_results.csv", index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到 weather_prediction_results.csv")