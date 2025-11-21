# HybridTriNet Streamlit App

- MERGE dữ liệu `du_lieu_noi_suy_clean` với file `price_petroleum` mới.
- Tự động bổ sung USD_Index (FRED) + GPRD.
- Huấn luyện mô hình HybridTriNet và dự báo `MG95, MG92, DO 0.001%, DO 0.05%`.

## Chạy app

```bash
cd oil-hypertrinet
streamlit run app/app_forecast.py
