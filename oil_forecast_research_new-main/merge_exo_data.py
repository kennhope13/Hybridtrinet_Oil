import pandas as pd
import os
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
print("🔄 Đang ghép nối Dữ liệu Ngoại sinh (GPR, USD, BRENT, WTI)...")

try:
    main_path = os.path.join(current_dir, 'data', 'processed', 'clean_data.csv')
    df_main = pd.read_csv(main_path)
    df_main['Ngày'] = pd.to_datetime(df_main['Ngày']).dt.normalize()
    df_main['YearMonth'] = df_main['Ngày'].dt.to_period('M')

    df_merged = df_main.copy()

    def load_fred_data(pattern, val_col_name, is_monthly=False):
        files = glob.glob(os.path.join(current_dir, 'data', 'raw', pattern))
        valid_files = [f for f in files if 'README' not in f.upper() and not os.path.basename(f).startswith('~$')]
        if not valid_files:
            return None
        
        filepath = valid_files[0]
        print(f"👉 Đang đọc {val_col_name} từ: {os.path.basename(filepath)}")
        
        df = pd.read_csv(filepath) if filepath.endswith('.csv') else pd.read_excel(filepath)
        date_col, val_col = df.columns[0], df.columns[1]
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce') 
        df = df.dropna(subset=[date_col])
        
        if is_monthly:
            df['YearMonth'] = df[date_col].dt.to_period('M')
            df = df.groupby('YearMonth')[val_col].mean().reset_index()
            df.rename(columns={val_col: val_col_name}, inplace=True)
        else:
            df = df[[date_col, val_col]].rename(columns={date_col: 'Ngày', val_col: val_col_name})
        return df

    # Tải USD Index
    df_usd = load_fred_data('DTWEXBGS*', 'USD_Index', False)
    if df_usd is not None: df_merged = pd.merge(df_merged, df_usd, on='Ngày', how='left')

    # Tải 4 biến Dầu thô mới
    df_brent_d = load_fred_data('DCOILBRENTEU*', 'Brent_EU_Daily', False)
    if df_brent_d is not None: df_merged = pd.merge(df_merged, df_brent_d, on='Ngày', how='left')

    df_wti_d = load_fred_data('DCOILWTICO*', 'WTI_Daily', False)
    if df_wti_d is not None: df_merged = pd.merge(df_merged, df_wti_d, on='Ngày', how='left')

    df_brent_m = load_fred_data('POILBREUSDM*', 'Brent_Global_Monthly', True)
    if df_brent_m is not None: df_merged = pd.merge(df_merged, df_brent_m, on='YearMonth', how='left')

    df_wti_m = load_fred_data('WTISPLC*', 'WTI_Monthly', True)
    if df_wti_m is not None: df_merged = pd.merge(df_merged, df_wti_m, on='YearMonth', how='left')

    # Tải GPR Daily
    gpr_files = glob.glob(os.path.join(current_dir, 'data', 'raw', 'data_gpr_daily_recent*'))
    valid_gpr = [f for f in gpr_files if 'README' not in f.upper() and not os.path.basename(f).startswith('~$')]
    if valid_gpr:
        gpr_path = valid_gpr[0]
        print(f"👉 Đang đọc GPR Daily từ: {os.path.basename(gpr_path)}")
        df_gpr = pd.read_csv(gpr_path) if gpr_path.endswith('.csv') else pd.read_excel(gpr_path)
        df_gpr['Ngày'] = pd.to_datetime(df_gpr['DAY'], format='%Y%m%d', errors='coerce').dt.normalize()
        df_gpr = df_gpr.rename(columns={'GPRD': 'GPR'}).dropna(subset=['Ngày'])
        df_merged = pd.merge(df_merged, df_gpr[['Ngày', 'GPR']], on='Ngày', how='left')

    # Dọn dẹp và Fill-forward cho cuối tuần/nghỉ lễ
    df_merged.drop(columns=['YearMonth'], inplace=True)
    fill_cols = ['USD_Index', 'Brent_EU_Daily', 'WTI_Daily', 'Brent_Global_Monthly', 'WTI_Monthly', 'GPR']
    for col in fill_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].ffill().bfill()

    # Xuất file
    output_path = os.path.join(current_dir, 'data', 'processed', 'clean_data_exo_ver1.csv')
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')

    count_exo = len([c for c in fill_cols if c in df_merged.columns])
    print(f"\n🎉 HOÀN TẤT! Đã ghép thành công {count_exo} biến ngoại sinh.")

except Exception as e:
    print(f"\n❌ LỖI: {e}")