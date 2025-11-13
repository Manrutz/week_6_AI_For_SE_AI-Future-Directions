
# simulation_notebook.py
# Synthetic sensor data generator and LightGBM yield prediction example.
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

def generate_synthetic(plot_id, days=120, seed=0):
    rng = np.random.RandomState(seed + plot_id)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    moisture = np.clip(30 + 10*np.sin(np.linspace(0,6.28,days)) + rng.normal(0,3,days), 5, 50)
    temp = np.clip(20 + 5*np.sin(np.linspace(0,3.14,days)) + rng.normal(0,2,days), 5, 40)
    rainfall = np.where(rng.rand(days) < 0.05, rng.uniform(0,20,days), 0)
    ndvi = np.clip(0.4 + 0.2*(moisture/50) + rng.normal(0,0.05,days), 0, 1)
    df = pd.DataFrame({'date': dates, 'plot_id': plot_id, 'moisture': moisture, 'temp': temp, 'rainfall': rainfall, 'ndvi': ndvi})
    # label: yield influenced by mean moisture, accumulated rainfall and mean ndvi
    label = 1000 + 30*df['moisture'].mean() + 200*df['ndvi'].mean() + rng.normal(0,50)
    return df, label

# generate dataset
dfs = []
labels = []
for pid in range(1,21):
    d, lab = generate_synthetic(pid, days=120, seed=42)
    dfs.append(d)
    labels.append({'plot_id': pid, 'yield': lab})
data = pd.concat(dfs, ignore_index=True)
labels_df = pd.DataFrame(labels)

# feature engineering: aggregate per plot
agg = data.groupby('plot_id').agg({
    'moisture': ['mean', 'std'],
    'temp': ['mean'],
    'rainfall': ['sum'],
    'ndvi': ['mean']
})
agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
df = agg.reset_index().merge(labels_df, on='plot_id')

# train LightGBM with GroupKFold by plot
X = df.drop(['plot_id','yield'], axis=1)
y = df['yield']
gkf = GroupKFold(n_splits=5)
maes = []
for tr_idx, val_idx in gkf.split(X, y, groups=df['plot_id']):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)
    params = {'objective':'regression','metric':'mae','verbosity':-1}
    bst = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=200, early_stopping_rounds=20)
    preds = bst.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    maes.append(mae)
print('CV MAE:', np.mean(maes))
