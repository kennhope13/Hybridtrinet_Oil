import pandas as pd
import numpy as np
import os

def clean_raw_data(input_path, output_path):
    print(f"📥 Đang đọc dữ liệu thô từ: {input_path}")
    
    # ---------------------------------------------------------
    # 💡 NÂNG CẤP: TỰ ĐỘNG ĐỌC FILE DÙ LÀ CSV HAY EXCEL
    # ---------------------------------------------------------
    try:
        if input_path.lower().endswith('.xlsx') or input_path.lower().endswith('.xls'):
            # Đọc file Excel (cần cài openpyxl)
            df = pd.read_excel(input_path)
        else:
            # Đọc file CSV
            df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return
    # ---------------------------------------------------------
    
    # 1. Loại bỏ các dòng rác chứa tiêu đề phụ hoặc không có Ngày
    df = df.dropna(subset=['Ngày'])
    df = df[df['Ngày'].astype(str).str.lower() != 'ngày']
    df = df[df['MG97'].astype(str).str.lower() != 'đơn vị tính']
    
    # 2. Xử lý cột thời gian
    df['Ngày'] = pd.to_datetime(df['Ngày'], errors='coerce')
    df = df.dropna(subset=['Ngày']).sort_values('Ngày').reset_index(drop=True)
    
    # 3. Lọc bỏ các cột rác (cột rỗng do Excel sinh ra)
    cols_to_keep = [c for c in df.columns if not str(c).startswith('Unnamed')]
    df = df[cols_to_keep]
    
    # 4. Ép kiểu số cho toàn bộ các cột giá (trừ cột Ngày)
    numeric_cols = [c for c in df.columns if c != 'Ngày']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 5. Lấp đầy dữ liệu bị thiếu bằng nội suy
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    df[numeric_cols] = df[numeric_cols].bfill().ffill() 
    
    # 6. Lưu ra file sạch luôn luôn ở định dạng CSV để AI đọc nhanh nhất
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Đã dọn dẹp xong! Giữ lại {len(df)} dòng.")
    print(f"💾 File sạch được lưu tại: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # ---------------------------------------------------------
    # ⚠️ CHÚ Ý: BẠN SỬA ĐUÔI FILE Ở ĐÂY CHO ĐÚNG THỰC TẾ TRONG THƯ MỤC RAW
    # ---------------------------------------------------------
    # Nếu file của bạn tên là "price_petroleum.xlsx", hãy viết như sau:
    RAW_FILE = os.path.join(project_root, 'data', 'raw', 'price_petroleum.xlsx')
    
    # (Nếu file tên là "price_petroleum.xlsx - Data.csv" thì bạn đổi lại nhé)
    
    CLEAN_FILE = os.path.join(project_root, 'data', 'processed', 'clean_data.csv')
    
    clean_raw_data(RAW_FILE, CLEAN_FILE)