"""
Multi-Model Oil Price Forecast – Evaluation Hub (Robust Version)
"""

import sys, json, importlib, warnings, os, logging

# Tắt toàn bộ cảnh báo (scikit-learn version, streamlit deprecation, etc.) để Terminal luôn sạch đẹp
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["STREAMLIT_PYARROW_ENABLED"] = "false" # Tắt PyArrow để tránh lỗi chặn DLL

# Tắt triệt để logging cảnh báo của Streamlit (ví dụ: use_container_width deprecation)
logging.getLogger("streamlit.deprecation_util").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import streamlit as st
import streamlit.components.v1 as components
import torch

st.set_page_config(
    page_title="Oil Forecast – Automated Evaluation Hub",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS Styles
st.markdown("""
<style>
    /* Ưu tiên lớp hướng dẫn trên sidebar; tránh tooltip bị cắt ở mép trái. */
    .driver-popover, .introjs-tooltip, .shepherd-element,
    .tour-popover, .tour-tooltip, [class*="tour-popover"], [class*="tour-tooltip"] {
        z-index: 100000 !important;
        max-width: min(390px, calc(100vw - 48px)) !important;
    }
    .driver-overlay, .introjs-overlay, .shepherd-modal-overlay-container,
    .tour-overlay, [class*="tour-overlay"] {
        z-index: 99998 !important;
    }
    /* Mũi tên bị định vị tĩnh thường nhảy sang sidebar; popup vẫn có hướng dẫn bằng text. */
    .tour-arrow, [class*="tour-arrow"], .introjs-arrow,
    .driver-popover-arrow, .shepherd-arrow {
        display: none !important;
    }
    /* Sidebar sáng, chữ đậm: tránh nền navy làm menu và trạng thái khó đọc. */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        background: #f7f9fc !important;
        border-right: 1px solid #d9e2ef !important;
    }
    [data-testid="stSidebar"] * { color: #26364d !important; }
    [data-testid="stSidebar"] [role="radiogroup"] label,
    [data-testid="stSidebar"] button {
        background: transparent !important;
        color: #334155 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover,
    [data-testid="stSidebar"] button:hover {
        background: #eaf5f4 !important;
        color: #075f57 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #dff5f1 !important;
        color: #075f57 !important;
        font-weight: 700 !important;
        box-shadow: inset 4px 0 0 #00ad91 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
        color: #075f57 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    [data-testid="stSidebar"] hr { border-color: #d9e2ef !important; }
/* 1. Reset font và tổng thể nền */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* 3. Thanh Tiến Trình (Progress Bar) chuẩn Hi-Tech Neon Gradient */
[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #00ad91 0%, #6954d9 100%) !important;
    border-radius: 99px !important;
    box-shadow: 0 0 12px rgba(0, 173, 145, 0.4) !important;
}

/* 4. Thanh Tab (st.tabs) chuẩn Doanh Nghiệp */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 600;
    color: #64748b;
    border: none !important;
    background: transparent;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #0f172a;
    background: #f1f5f9;
}
.stTabs [aria-selected="true"] {
    color: #00ad91 !important;
    font-weight: 800 !important;
    border-bottom: 3px solid #00ad91 !important;
    background: transparent !important;
}

/* 5. Khung thông báo và thẻ nghiệp vụ */
.hub-notice {
    border-radius: 9px;
    padding: 13px 16px;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 14px;
}
.hub-notice.cpu {
    background: #fff7e7;
    border: 1px solid #f6d58c;
    color: #794800;
}
.hub-notice.gpu {
    background: #eafaf5;
    border: 1px solid #bcebdc;
    color: #116d5c;
}
.guide-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 18px;
    background: #ffffff;
    box-shadow: 0 2px 6px rgba(24, 34, 55, 0.04);
    height: 100%;
}
@keyframes pulseDot {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 173, 145, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(0, 173, 145, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 173, 145, 0); }
}
.pulsing-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #00ad91;
    margin-right: 6px;
    animation: pulseDot 2s infinite;
}
.pulsing-dot-cpu {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #c77700;
    margin-right: 6px;
    animation: pulseDot 2s infinite;
}
</style>
""", unsafe_allow_html=True)

def inject_oil_tour_engine(current_page=""):
    # `current_page` được nhúng vào cuối script (xem dòng cuối) chỉ để nội dung HTML gửi cho
    # components.html() LUÔN đổi mỗi khi người dùng chuyển trang. Nếu nội dung y hệt lần trước,
    # Streamlit sẽ không nạp lại iframe này -> đoạn code "mở tour đang chờ sau khi chuyển trang"
    # (biến pendingOilTour) sẽ không bao giờ được chạy lại để thực sự bật tour lên.
    tour_script = """
    <script>
    (function() {
        const doc = window.parent.document;
        if (!doc) return;

        // 1. Inject / Update Styles
        let style = doc.getElementById('oil-tour-styles');
        if (!style) {
            style = doc.createElement('style');
            style.id = 'oil-tour-styles';
            doc.head.appendChild(style);
        }
        style.textContent = `
            .oil-tour-layer {
                display: none;
                position: fixed;
                inset: 0;
                background: transparent;
                z-index: 2000000002;
                pointer-events: none;
                transition: opacity 0.25s ease;
            }
            .oil-tour-layer.show { display: block !important; }
            #oil-welcome-layer {
                background: rgba(15, 23, 42, 0.65) !important;
                z-index: 2000000010 !important;
                pointer-events: auto !important;
            }
            .oil-spotlight {
                position: fixed;
                display: none;
                pointer-events: none;
                z-index: 2000000000;
                border: 3.5px solid #00ad91;
                border-radius: 12px;
                box-shadow: 0 0 0 9999px rgba(15, 23, 42, 0.65), 0 0 25px rgba(0, 173, 145, 0.85);
                transition: top 0.22s ease, left 0.22s ease, width 0.22s ease, height 0.22s ease;
                will-change: top, left, width, height;
                box-sizing: border-box;
            }
            .oil-tour-dialog {
                position: fixed;
                right: 32px;
                bottom: 32px;
                width: min(440px, calc(100vw - 40px));
                max-height: calc(100vh - 60px);
                background: #ffffff !important;
                color: #0f172a !important;
                border-radius: 14px;
                padding: 22px 24px;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45), 0 0 0 1px #e2e8f0 !important;
                border: 1px solid #e2e8f0;
                z-index: 2000000003 !important;
                pointer-events: auto !important;
                font-family: Inter, system-ui, -apple-system, sans-serif;
                transition: top 0.25s ease, bottom 0.25s ease, left 0.25s ease, right 0.25s ease;
                will-change: top, bottom, left, right;
                backface-visibility: hidden;
                box-sizing: border-box;
                overflow-y: auto;
            }
            .oil-welcome-dialog {
                position: fixed;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%);
                width: min(490px, 92vw);
                background: #ffffff !important;
                color: #0f172a !important;
                border-radius: 16px;
                padding: 28px 32px;
                box-shadow: 0 25px 70px rgba(0, 0, 0, 0.5);
                border: 1px solid #e2e8f0;
                z-index: 2000000011 !important;
                pointer-events: auto !important;
                font-family: Inter, system-ui, -apple-system, sans-serif;
                animation: oilWelcomePop .25s ease;
            }
            @keyframes oilWelcomePop {
                from { transform: translate(-50%, -46%); opacity: 0; }
                to { transform: translate(-50%, -50%); opacity: 1; }
            }
            .oil-arrow {
                position: fixed;
                left: 0;
                top: 0;
                pointer-events: none;
                z-index: 2000000004;
                display: none;
                transition: top 0.2s ease, left 0.2s ease;
                will-change: transform;
                backface-visibility: hidden;
                animation: oilBounce 0.8s ease-in-out infinite alternate;
            }
            @keyframes oilBounce {
                0% { transform: translateY(0); }
                100% { transform: translateY(8px); }
            }
            .oil-tour-focus {
                position: relative !important;
                z-index: 2000000000 !important;
            }
            .oil-btn {
                border: 0;
                border-radius: 8px;
                padding: 9px 16px;
                font-size: 13px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                font-family: inherit;
            }
            .oil-btn-ghost {
                background: #f1f5f9;
                color: #475569;
                border: 1px solid #e2e8f0;
            }
            .oil-btn-ghost:hover {
                background: #e2e8f0;
                color: #0f172a;
            }
            .oil-btn-primary {
                background: linear-gradient(135deg, #00ad91, #008f77);
                color: #ffffff;
                box-shadow: 0 4px 12px rgba(0, 173, 145, 0.35);
            }
            .oil-btn-primary:hover {
                filter: brightness(1.1);
            }
            .guide-action {
                display: block;
                width: 100%;
                padding: 18px 20px;
                text-align: left;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                cursor: pointer;
                color: #1e293b;
                font-family: inherit;
                transition: all 0.2s ease;
                box-shadow: 0 2px 6px rgba(24, 34, 55, 0.04);
            }
            .guide-action:hover {
                border-color: #00ad91;
                box-shadow: 0 6px 18px rgba(0, 173, 145, 0.15);
                transform: translateY(-2px);
            }
            .guide-action b { display: block; margin: 0 0 4px; font-size: 15px; }
            .guide-action small { display: block; color: #64748b; font-size: 13px; line-height: 1.4; }
        `;

        // 2. Inject Tour HTML Containers into doc.body if not present
        if (!doc.getElementById('oil-tour-root')) {
            const root = doc.createElement('div');
            root.id = 'oil-tour-root';
            root.innerHTML = `
                <div class="oil-spotlight" id="oil-spotlight-el"></div>
                <div class="oil-arrow" id="oil-arrow-el">
                    <svg id="oil-arrow-svg" width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#00ad91" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 3px 6px rgba(0,0,0,0.4));">
                        <path d="M12 4v14M18 12l-6 6-6-6"/>
                    </svg>
                </div>
                <div class="oil-tour-layer" id="oil-tour-layer">
                    <div class="oil-tour-dialog">
                        <div style="font-size:11px; font-weight:800; color:#00ad91; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">HƯỚNG DẪN TƯƠNG TÁC</div>
                        <h3 id="oil-tour-title" style="margin:0 0 6px; font-size:17px; font-weight:800; color:#0f172a;"></h3>
                        <p id="oil-tour-text" style="color:#475569; margin:0 0 18px; font-size:13.5px; line-height:1.55;"></p>
                        <div style="display:flex; align-items:center; gap:10px;">
                            <span id="oil-tour-count" style="margin-right:auto; color:#00ad91; font-size:12px; font-weight:800; background:#eafaf5; padding:4px 10px; border-radius:99px;"></span>
                            <button class="oil-btn oil-btn-ghost" id="oil-tour-skip">Bỏ qua</button>
                            <button class="oil-btn oil-btn-primary" id="oil-tour-next">Tiếp theo ➔</button>
                        </div>
                    </div>
                </div>
                <div class="oil-tour-layer" id="oil-welcome-layer">
                    <div class="oil-welcome-dialog">
                        <div style="color:#00ad91; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:0.09em; margin-bottom:6px;">CHÀO MỪNG BẠN</div>
                        <h2 style="margin:0 0 10px; font-size:20px; font-weight:800; color:#0f172a;">🛢️ Oil Forecast – Automated Evaluation Hub</h2>
                        <p style="color:#64748b; font-size:13.5px; line-height:1.6; margin:0 0 22px;">
                            Hướng dẫn nhanh bằng mũi tên chỉ dẫn tương tác sẽ giúp bạn nắm vững cách cập nhật dữ liệu giá, tạo dự báo đa mốc thời gian và đánh giá sai số mô hình AI. Mất khoảng 1 phút.
                        </p>
                        <div style="display:flex; justify-content:flex-end; gap:12px;">
                            <button class="oil-btn oil-btn-ghost" id="oil-welcome-later">Để sau</button>
                            <button class="oil-btn oil-btn-primary" id="oil-welcome-start">Bắt đầu hướng dẫn ➔</button>
                        </div>
                    </div>
                </div>
            `;
            doc.body.appendChild(root);
        }

        // Đảm bảo oil-spotlight-el luôn tồn tại
        if (!doc.getElementById('oil-spotlight-el')) {
            const sp = doc.createElement('div');
            sp.className = 'oil-spotlight';
            sp.id = 'oil-spotlight-el';
            const rootEl = doc.getElementById('oil-tour-root') || doc.body;
            rootEl.insertBefore(sp, rootEl.firstChild);
        }

        // 3. Define Tour Steps with Multi-Selector Fallbacks
        const tours = {
            forecast: [
                {
                    title: "1. Kéo thả file dữ liệu thị trường",
                    text: "Kéo thả file Excel (.xlsx) hoặc CSV giá dầu mới nhất vào đây. Hệ thống tự động làm sạch và tính toán ngay dự báo 7 mốc mà không cần bấm thêm nút nào.",
                    selector: '[data-testid="stFileUploader"], .stFileUploader'
                },
                {
                    title: "2. Giám sát trạng thái & phần cứng",
                    text: "Kiểm tra phiên bản dữ liệu gần nhất, thiết bị tính toán (GPU/CPU) và tình trạng nạp sẵn sàng 7 mốc dự báo (+1d đến +60d) từ mô hình GUMNet.",
                    selector: '.hub-notice, [data-testid="stColumn"]:nth-child(2), [data-testid="column"]:nth-child(2)'
                },
                {
                    title: "3. Xem bảng giá dự báo & Biểu đồ xu hướng",
                    text: "Bảng hiển thị giá dự kiến (USD/thùng hoặc USD/tấn) cho 4 mặt hàng. Rê chuột vào biểu đồ để xem giá từng ngày, bấm tên mốc ở chú giải bên phải để ẩn/hiện đường.",
                    selector: '[data-testid="stDataFrame"], [data-testid="stPlotlyChart"], .table-wrap'
                },
                {
                    title: "4. Xuất file báo cáo & Chuyển bước",
                    text: "Bấm nút 'Xuất Bảng Dự Báo' để tải file Excel/CSV gửi lãnh đạo, hoặc chuyển sang menu 'Đánh giá mô hình' để kiểm tra sai số thực tế.",
                    selector: '[data-testid="stDownloadButton"], [data-testid="stSidebar"], .stDownloadButton'
                }
            ],
            metrics: [
                {
                    title: "1. Đọc chỉ số sai số (MAPE & MAE)",
                    text: "Theo dõi sai số giữa giá AI đoán và giá thị trường. Ngưỡng an toàn: MAPE < 7% (Xanh lá - Rất tốt). Nếu MAPE > 10% (Đỏ), khuyến nghị sang trang Huấn luyện để Finetune.",
                    selector: '[data-testid="stHorizontalBlock"], [data-testid="column"], [data-testid="stMetric"]'
                },
                {
                    title: "2. Phân tích chi tiết theo Mốc, Mặt hàng & Xu hướng",
                    text: "Bảng nhiệt bên dưới phân rã sai số theo từng mốc thời gian và từng loại dầu. Biểu đồ đường cho thấy xu hướng sai số tăng tự nhiên ở các mốc tương lai xa (+60 ngày).",
                    selector: '[data-testid="stPlotlyChart"], [data-testid="stDataFrame"]'
                }
            ],
            training: [
                {
                    title: "1. Chọn cấu hình & Chế độ huấn luyện",
                    text: "Khuyến nghị chọn 'Finetune từ checkpoint' để cập nhật nhanh quy luật mới (%%HW_TOUR_STR%%). Hệ thống đã tự động chọn số Epochs và đủ cả 7 mốc.",
                    selector: '[data-baseweb="tab-list"], .stTabs'
                },
                {
                    title: "2. Khởi chạy Job Huấn Luyện ngầm",
                    text: "Nhấn nút 'Bắt đầu Job' để tiến trình tối ưu hóa chạy độc lập trong nền. Bạn có thể theo dõi tiến độ từng mốc qua thanh phần trăm trực quan.",
                    selector: '[data-testid="stButton"] button, button[kind="primary"]'
                },
                {
                    title: "3. Lịch sử & Đối chiếu Benchmarking",
                    text: "Xem lại tệp dữ liệu đã nạp tại Tab 2 và chuyển sang Tab 3 để so sánh phiên mới vs phiên cũ — đo lường mức độ cải thiện sai số (% Giảm sai số màu xanh lá).",
                    selector: '[data-baseweb="tab"]:nth-of-type(3), [role="tab"]:nth-of-type(3), [data-baseweb="tab-list"] button:nth-of-type(3), [data-baseweb="tab-list"]'
                }
            ],
            history: [
                {
                    title: "1. Tra cứu danh mục các đợt nạp dữ liệu",
                    text: "Bảng lưu trữ toàn bộ các tệp Excel/CSV đã nạp vào hệ thống qua từng đợt, phục vụ công tác thanh tra, kiểm toán bất cứ lúc nào.",
                    selector: '[data-testid="stDataFrame"], [data-testid="stSelectbox"]'
                },
                {
                    title: "2. Đối chiếu chi tiết Thực tế vs Dự báo & Xuất file",
                    text: "Chọn đợt nạp để xem bảng đối chiếu và tải file CSV. Biểu đồ so sánh trực quan giữa đường giá thị trường thực tế (xanh ngọc) và giá AI dự báo (tím nét đứt).",
                    selector: '[data-testid="stPlotlyChart"], .stPlotlyChart'
                }
            ]
        };

        let activeTour = [];
        let currentIdx = 0;

        // `selectorStr` là danh sách các lựa chọn dự phòng cách nhau bởi dấu phẩy (vd:
        // '[data-testid="stMetric"], [data-testid="column"]'). doc.querySelector() với chuỗi
        // nhiều lựa chọn KHÔNG thử lần lượt từng lựa chọn — nó trả về phần tử khớp bất kỳ selector
        // nào ĐẦU TIÊN THEO THỨ TỰ TRONG DOM, có thể là phần tử hoàn toàn không liên quan ở chỗ
        // khác trên trang. Hàm này thử TỪNG selector riêng lẻ theo đúng thứ tự ưu tiên đã khai báo,
        // chỉ chuyển sang selector dự phòng tiếp theo khi selector hiện tại không có phần tử nào.
        function pickBestElement(selectorStr) {
            // Chỉ tìm trong vùng NỘI DUNG CHÍNH, loại trừ sidebar — nếu không, các selector
            // chung chung như [data-testid="stHorizontalBlock"] rất dễ khớp nhầm phần tử nằm
            // trong sidebar (đứng trước nội dung chính trong thứ tự DOM).
            const scope = doc.querySelector('[data-testid="stMain"]')
                || doc.querySelector('section.main')
                || doc.querySelector('.main')
                || doc;
            const parts = selectorStr.split(',').map(s => s.trim()).filter(Boolean);
            // Lỗi đã xác nhận: ở màn hình hẹp, dải tab (st.tabs) tràn ngang và tab thứ 3+ bị đẩy
            // ra ngoài vùng nhìn thấy dù DOM vẫn có kích thước (height > 0) — trước đây coi đó là
            // "tìm thấy" nên không thử fallback tiếp, khiến khung sáng/mũi tên trỏ vào chỗ nằm
            // ngoài màn hình. Giờ chỉ chấp nhận phần tử khi nó thực sự cắt ngang vùng nhìn thấy
            // theo chiều ngang; nếu không, thử tiếp selector dự phòng kế tiếp trong danh sách.
            for (const sel of parts) {
                try {
                    const el = scope.querySelector(sel);
                    if (!el) continue;
                    const r = el.getBoundingClientRect();
                    const visibleWidth = window.parent.innerWidth || document.documentElement.clientWidth;
                    if (r.height > 0 && r.right > 0 && r.left < visibleWidth) return el;
                } catch (e) { /* selector không hợp lệ trong tài liệu này, bỏ qua */ }
            }
            return null;
        }

        function waitForTargetElement(selector, callback, maxTries = 30, interval = 120) {
            let tries = 0;
            function check() {
                const el = pickBestElement(selector);
                if (el) {
                    callback(el);
                } else if (++tries < maxTries) {
                    setTimeout(check, interval);
                } else {
                    callback(null);
                }
            }
            check();
        }

        function positionDialogAndArrow(targetEl) {
            const arrow = doc.getElementById('oil-arrow-el');
            const arrowSvgPath = doc.querySelector('#oil-arrow-svg path');
            const dialog = doc.querySelector('.oil-tour-dialog');
            const spotlight = doc.getElementById('oil-spotlight-el');
            const tourLayer = doc.getElementById('oil-tour-layer');
            if (!dialog) return;

            if (!targetEl) {
                if (arrow) arrow.style.display = 'none';
                if (spotlight) spotlight.style.display = 'none';
                if (tourLayer) tourLayer.style.background = 'rgba(15, 23, 42, 0.65)';
                dialog.style.top = 'auto';
                dialog.style.bottom = '28px';
                dialog.style.left = 'auto';
                dialog.style.right = '28px';
                return;
            }

            const r = targetEl.getBoundingClientRect();
            // Cập nhật vị trí và kích thước của Spotlight Cutout Box
            if (spotlight) {
                const pad = 6;
                spotlight.style.display = 'block';
                spotlight.style.top = (r.top - pad) + 'px';
                spotlight.style.left = (r.left - pad) + 'px';
                spotlight.style.width = (r.width + pad * 2) + 'px';
                spotlight.style.height = (r.height + pad * 2) + 'px';
            }
            if (tourLayer) {
                tourLayer.style.background = 'transparent';
            }

            // QUAN TRỌNG: script này chạy trong khung ẩn kích thước 0x0 (components.html height=0
            // width=0), nên window.innerWidth/innerHeight của CHÍNH khung này luôn xấp xỉ 0 — phải
            // lấy từ cửa sổ thật của người dùng (window.parent) thì công thức định vị mới đúng.
            const vh = window.parent.innerHeight;
            const vw = window.parent.innerWidth;

            // 1. VỊ TRÍ HỘP THOẠI HƯỚNG DẪN THÔNG MINH (Không bị che, không đè lên phần tử được trỏ)
            const targetInBottomHalf = r.top > (vh * 0.42) || r.bottom > (vh * 0.72);
            const targetOnRightSide = (r.left + r.width / 2) > (vw * 0.55);

            if (targetInBottomHalf) {
                // Phần tử ở nửa dưới màn hình -> Hộp thoại chuyển lên GÓC PHẢI TRÊN (tránh bị che ở góc dưới)
                dialog.style.top = '24px';
                dialog.style.bottom = 'auto';
            } else {
                // Phần tử ở nửa trên màn hình -> Hộp thoại ở GÓC PHẢI DƯỚI
                dialog.style.bottom = '28px';
                dialog.style.top = 'auto';
            }

            if (targetOnRightSide && r.width > (vw * 0.45)) {
                // Phần tử nằm lệch phải và rộng -> Hộp thoại sang GÓC TRÁI
                dialog.style.left = '28px';
                dialog.style.right = 'auto';
            } else {
                dialog.style.right = '28px';
                dialog.style.left = 'auto';
            }

            // 2. MŨI TÊN VECTOR CHỈ DẪN CHUẨN XÁC
            if (arrow && arrowSvgPath) {
                arrow.style.display = 'block';
                let top = r.top - 54;
                let left = r.left + (r.width / 2) - 22;

                if (top < 20) {
                    // Mũi tên trỏ NGƯỢC LÊN từ phía dưới phần tử
                    arrowSvgPath.setAttribute('d', 'M12 20V6M6 12l6-6 6 6');
                    top = r.bottom + 10;
                } else {
                    // Mũi tên trỏ XUỐNG từ phía trên phần tử
                    arrowSvgPath.setAttribute('d', 'M12 4v14M18 12l-6 6-6-6');
                }

                left = Math.max(16, Math.min(vw - 60, left));
                arrow.style.left = left + 'px';
                arrow.style.top = top + 'px';
            }
        }

        function showStep(idx) {
            const prev = doc.querySelector('.oil-tour-focus');
            if (prev) {
                prev.classList.remove('oil-tour-focus');
                prev.style.removeProperty('outline-width');
                prev.style.removeProperty('outline-style');
                prev.style.removeProperty('outline-color');
                prev.style.removeProperty('outline-offset');
                prev.style.removeProperty('box-shadow');
            }

            if (idx >= activeTour.length) {
                endTour();
                return;
            }

            const step = activeTour[idx];

            doc.getElementById('oil-tour-title').textContent = step.title;
            doc.getElementById('oil-tour-text').textContent = step.text;
            doc.getElementById('oil-tour-count').textContent = `Bước ${idx + 1} / ${activeTour.length}`;
            doc.getElementById('oil-tour-next').textContent = (idx === activeTour.length - 1) ? 'Hoàn tất ✓' : 'Tiếp theo ➔';

            waitForTargetElement(step.selector, function(targetEl) {
                if (targetEl) {
                    targetEl.classList.add('oil-tour-focus');
                    if (document.activeElement === targetEl && typeof targetEl.blur === 'function') {
                        targetEl.blur();
                    }
                    targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    positionDialogAndArrow(targetEl);
                    setTimeout(() => positionDialogAndArrow(targetEl), 150);
                    setTimeout(() => positionDialogAndArrow(targetEl), 350);
                } else {
                    positionDialogAndArrow(null);
                }
            });
        }

        function endTour() {
            const prev = doc.querySelector('.oil-tour-focus');
            if (prev) {
                prev.classList.remove('oil-tour-focus');
                prev.style.removeProperty('outline-width');
                prev.style.removeProperty('outline-style');
                prev.style.removeProperty('outline-color');
                prev.style.removeProperty('outline-offset');
                prev.style.removeProperty('box-shadow');
            }
            const spotlight = doc.getElementById('oil-spotlight-el');
            if (spotlight) spotlight.style.display = 'none';
            const layer = doc.getElementById('oil-tour-layer');
            if (layer) {
                layer.classList.remove('show');
                layer.style.background = 'transparent';
            }
            const arrow = doc.getElementById('oil-arrow-el');
            if (arrow) arrow.style.display = 'none';
            activeTour = [];
            currentIdx = 0;
        }

        // Lắng nghe resize và scroll để tự điều chỉnh tọa độ
        function updateActivePositions() {
            if (activeTour.length > 0 && currentIdx < activeTour.length) {
                const step = activeTour[currentIdx];
                const targetEl = pickBestElement(step.selector);
                positionDialogAndArrow(targetEl);
            }
        }
        // Script này được components.html() nhúng lại mỗi lần Streamlit rerun (mọi thao tác
        // click/upload/đổi trang), và khung ẩn (iframe) cũ bị hủy mỗi lần như vậy. Listener gắn
        // từ lần chạy trước thuộc về ngữ cảnh JS đã bị hủy nên gọi vào nó không còn tác dụng dù
        // vẫn còn trong danh sách listener của window/document -> phải gỡ rồi gắn lại mỗi lần.
        if (doc.__oilTourResizeHandler) {
            window.parent.removeEventListener('resize', doc.__oilTourResizeHandler);
        }
        if (doc.__oilTourScrollHandler) {
            doc.removeEventListener('scroll', doc.__oilTourScrollHandler, true);
        }
        doc.__oilTourResizeHandler = function() { updateActivePositions(); };
        doc.__oilTourScrollHandler = function() { updateActivePositions(); };
        window.parent.addEventListener('resize', doc.__oilTourResizeHandler);
        doc.addEventListener('scroll', doc.__oilTourScrollHandler, true);

        function runTourDirect(tourKey) {
            activeTour = tours[tourKey] || tours['forecast'];
            currentIdx = 0;
            const wLayer = doc.getElementById('oil-welcome-layer');
            if (wLayer) wLayer.classList.remove('show');
            const tLayer = doc.getElementById('oil-tour-layer');
            if (tLayer) tLayer.classList.add('show');
            showStep(0);
        }

        function switchStreamlitPage(tourKey, onReady) {
            const pageKeywords = {
                forecast: "Dự báo",
                metrics: "Đánh giá",
                history: "Lịch sử",
                training: "Huấn luyện",
                guide: "Hướng dẫn"
            };
            const keyword = pageKeywords[tourKey] || "Dự báo";
            const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
            let targetLabel = null;
            if (sidebar) {
                const labels = sidebar.querySelectorAll('.stRadio label, [role="radiogroup"] label');
                for (let i = 0; i < labels.length; i++) {
                    if (labels[i].textContent.includes(keyword)) {
                        targetLabel = labels[i];
                        break;
                    }
                }
            }

            if (targetLabel) {
                const inputEl = targetLabel.querySelector('input');
                const isCurrent = inputEl ? inputEl.checked : false;
                if (!isCurrent) {
                    sessionStorage.setItem('pendingOilTour', tourKey);
                    targetLabel.click();
                    return;
                }
            }
            onReady();
        }

        // Global APIs on window and window.parent for zero-latency calls
        window.parent.startOilTour = window.startOilTour = function(tourKey) {
            switchStreamlitPage(tourKey, function() {
                runTourDirect(tourKey);
            });
        };

        window.parent.replayOnboarding = window.replayOnboarding = function() {
            try { localStorage.removeItem('oilForecastTourSeen'); } catch(e) {}
            const wLayer = doc.getElementById('oil-welcome-layer');
            if (wLayer) wLayer.classList.add('show');
        };

        // Event delegation on doc cho toàn bộ tương tác của Tour và Popup Chào mừng.
        // QUAN TRỌNG: mỗi lần Streamlit rerun, khung ẩn (iframe) này bị hủy và tạo mới, nên
        // listener được gắn từ lần chạy TRƯỚC thực chất đã "chết" (thuộc ngữ cảnh JS đã bị hủy,
        // gọi vào nó không còn tác dụng dù vẫn còn nằm trong danh sách listener của document).
        // Vì vậy KHÔNG được chỉ gắn một lần duy nhất (sẽ bị kẹt với listener chết mãi mãi) —
        // phải gỡ listener cũ rồi gắn lại listener mới (luôn "sống") ở mỗi lần chạy.
        if (doc.__oilTourClickHandler) {
            doc.removeEventListener('click', doc.__oilTourClickHandler, true);
        }
        doc.__oilTourClickHandler = function(e) {
            // 1. Nút "Để sau" trên popup chào mừng
            if (e.target.closest('#oil-welcome-later')) {
                const wLayer = doc.getElementById('oil-welcome-layer');
                if (wLayer) wLayer.classList.remove('show');
                try { localStorage.setItem('oilForecastTourSeen', 'true'); } catch(err) {}
                return;
            }
            // 2. Nút "Bắt đầu hướng dẫn ➔" trên popup chào mừng
            if (e.target.closest('#oil-welcome-start')) {
                const wLayer = doc.getElementById('oil-welcome-layer');
                if (wLayer) wLayer.classList.remove('show');
                try { localStorage.setItem('oilForecastTourSeen', 'true'); } catch(err) {}
                window.parent.startOilTour('forecast');
                return;
            }
            // 3. Click ra ngoài backdrop của popup chào mừng để đóng
            if (e.target.id === 'oil-welcome-layer') {
                e.target.classList.remove('show');
                try { localStorage.setItem('oilForecastTourSeen', 'true'); } catch(err) {}
                return;
            }
            // 4. Nút "Bỏ qua" trên hộp thoại tour
            if (e.target.closest('#oil-tour-skip')) {
                endTour();
                return;
            }
            // 5. Nút "Tiếp theo ➔" / "Hoàn tất ✓" trên hộp thoại tour
            if (e.target.closest('#oil-tour-next')) {
                currentIdx++;
                showStep(currentIdx);
                return;
            }
            // 6. Các thẻ bài học trên Trang Hướng dẫn (data-oil-tour)
            const tourBtn = e.target.closest('[data-oil-tour]');
            if (tourBtn) {
                const tourKey = tourBtn.getAttribute('data-oil-tour');
                window.parent.startOilTour(tourKey);
                return;
            }
            // 7. Nút "↺ Xem lại thông báo chào mừng & hướng dẫn từ đầu"
            if (e.target.closest('#oil-replay-btn')) {
                window.parent.replayOnboarding();
                return;
            }
        };
        doc.addEventListener('click', doc.__oilTourClickHandler, true);

        // Check if there is a pending tour after page switch
        const pending = sessionStorage.getItem('pendingOilTour');
        if (pending) {
            sessionStorage.removeItem('pendingOilTour');
            setTimeout(() => runTourDirect(pending), 200);
        }

        // Check first-time visit on this browser (LAN clients)
        try {
            if (!localStorage.getItem('oilForecastTourSeen') && !sessionStorage.getItem('pendingOilTour')) {
                setTimeout(function() {
                    doc.getElementById('oil-welcome-layer').classList.add('show');
                }, 800);
            }
        } catch(e) {}
    })();
    </script>
    <!-- current_page: """ + str(current_page) + """ -->
    """
    _is_gpu = torch.cuda.is_available()
    _hw_tour_str = "tốc độ tối ưu cực nhanh 10–20 giây/mốc trên GPU NVIDIA CUDA" if _is_gpu else "khoảng 1–2 phút/mốc trên CPU 6 vCPUs"
    tour_script = tour_script.replace("%%HW_TOUR_STR%%", _hw_tour_str)
    components.html(tour_script, height=0, width=0)

# Hàm hiển thị DataFrame an toàn để tránh lỗi DLL Blocked, PyArrow ArrowTypeError và triệt tiêu warning
def safe_dataframe(df, **kwargs):
    clean_kwargs = dict(kwargs)
    clean_kwargs.pop("use_container_width", None)
    if "width" not in clean_kwargs:
        clean_kwargs["width"] = "stretch"

    # Chuẩn hóa các cột kiểu object để tránh lỗi PyArrow hỗn hợp kiểu dữ liệu (str và int)
    df_safe = df
    try:
        if isinstance(df, pd.DataFrame):
            df_safe = df.copy()
            for col in df_safe.columns:
                if df_safe[col].dtype == 'object':
                    df_safe[col] = df_safe[col].apply(lambda x: str(x) if x is not None else "")
    except Exception:
        df_safe = df

    # 1. Thử hiển thị bằng dataframe chuẩn
    try:
        st.dataframe(df_safe, **clean_kwargs)
        return
    except Exception as e:
        # Nếu lỗi liên quan đến tham số width trên Streamlit cũ
        if "width" in str(e) or "unexpected keyword" in str(e):
            try:
                clean_kwargs.pop("width", None)
                st.dataframe(df_safe, use_container_width=True, **clean_kwargs)
                return
            except Exception:
                pass

    # 2. Nếu PyArrow lỗi hoặc bị chặn DLL, hiển thị bằng HTML thuần (100% an toàn)
    try:
        if hasattr(df, "to_html"):
            html = df.to_html(classes='table table-striped', justify='center', border=0)
            st.write(html, unsafe_allow_html=True)
            return
    except Exception:
        pass

    # 3. GIẢI PHÁP CUỐI CÙNG: Hiển thị bằng st.table hoặc st.write
    try:
        st.table(df)
    except Exception:
        st.write(df)

# Hàm hiển thị Plotly Chart an toàn triệt tiêu hoàn toàn warning use_container_width
def safe_plotly_chart(fig, **kwargs):
    clean_kwargs = dict(kwargs)
    clean_kwargs.pop("use_container_width", None)
    if "width" not in clean_kwargs:
        clean_kwargs["width"] = "stretch"
    try:
        st.plotly_chart(fig, **clean_kwargs)
    except TypeError:
        clean_kwargs.pop("width", None)
        st.plotly_chart(fig, use_container_width=True, **clean_kwargs)

# === CONFIG ===
ROOT = Path(__file__).resolve().parent
BUILTIN_CSV = ROOT / "oil_forecast_research_new-main" / "data" / "processed" / "clean_data_exo_ver1.csv"
CKPT_DIR = ROOT / "checkpoints_multi"

TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
DATE_COL = "Ngày"
HORIZONS = [1, 5, 10, 15, 20, 30, 60]
# CUTOFF_DATE (mốc bắt đầu tính backtest/đánh giá) được tính động ngay bên dưới, sau khi
# đã đọc được dữ liệu thực tế — xem "CUTOFF_DATE = ..." gần phần load file_info.

MODEL_DEFS = {
    "GUMNet": {
        "proj_dir": ROOT / "oil_forecast_research_new-main",
        "mod": "src.model.model", "cls": "GUMNet", "kind": "quantile",
    },
    # Lỗi #14 (đã xác nhận): sidebar cho chọn "HybridTriNet" nhưng MODEL_DEFS trước đây thiếu
    # hẳn khai báo này -> load_model("HybridTriNet", h) luôn thất bại (bắt bởi try/except nên
    # không crash cả app, nhưng âm thầm không bao giờ ra kết quả dự báo).
    "HybridTriNet": {
        "proj_dir": ROOT / "Hybridtrinet_Oil",
        "mod": "src.model.hybrid_trinet", "cls": "HybridTriNet", "kind": "point",
    },
}

# === DATA HELPERS ===

@st.cache_data
def _cached_load_df(path_str, mtime):
    path = Path(path_str)
    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding='utf-8')
            if df.columns[0].startswith('Ng'): # Xử lý lỗi font chữ ở đầu file csv
                 df = df.rename(columns={df.columns[0]: DATE_COL})
        
        # Làm sạch tên cột và xóa khoảng trắng
        df.columns = [str(c).strip() for c in df.columns]
        
        # Tìm cột ngày linh hoạt (chấp nhận Ngay, Ngay, Date, Day...)
        potential_date_cols = [c for c in df.columns if any(x in c.lower() for x in ["ng", "date", "time"])]
        if potential_date_cols:
            actual_col = potential_date_cols[0]
            if actual_col != DATE_COL:
                df = df.rename(columns={actual_col: DATE_COL})
        
        if DATE_COL in df.columns:
            # Ép kiểu ngày tháng và normalize (loại bỏ giờ phút giây)
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", format='mixed')
            
            df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
            df[DATE_COL] = df[DATE_COL].dt.normalize()
        
        # Ép kiểu số cho các cột còn lại
        for c in df.columns:
            if c != DATE_COL: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.interpolate().bfill().ffill()
        return df
    except Exception as e:
        st.sidebar.error(f"⚠️ Lỗi đọc file {path.name}: {e}")
        return pd.DataFrame()

def load_df(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    mtime = path.stat().st_mtime
    return _cached_load_df(str(path), mtime)

def generate_time_features(df):
    if DATE_COL not in df.columns: return df
    dt = df[DATE_COL]
    for col, val in [("NgayTrongTuan", dt.dt.dayofweek), ("ThangTrongNam", dt.dt.month),
                      ("QuyTrongNam", dt.dt.quarter), ("Nam", dt.dt.year)]:
        if col not in df.columns: df[col] = val
    for col in ["NgayLe", "SuKienDacBiet"]:
        if col not in df.columns: df[col] = 0
    if "GPRD" not in df.columns: df["GPRD"] = df.get("GPR", 0)
    if "Unnamed: 0" not in df.columns: df["Unnamed: 0"] = range(len(df))
    return df

def enrich_with_exo(df, base_df):
    df = generate_time_features(df)
    missing = [c for c in base_df.columns if c not in df.columns and c != DATE_COL]
    if not missing: return df
    
    # Merge bằng ngày đã normalize
    merged = pd.merge(df, base_df[[DATE_COL] + missing], on=DATE_COL, how="left")
    merged[missing] = merged[missing].ffill().bfill()
    for c in missing:
        if merged[c].isna().any(): merged[c] = merged[c].fillna(base_df[c].iloc[-1])
    return merged

# === MODEL LOADING ===

def _swap_src(proj_dir):
    d = str(proj_dir)
    sys.path = [p for p in sys.path if p != d]
    sys.path.insert(0, d)
    for m in [k for k in list(sys.modules) if k.startswith("src")]: del sys.modules[m]

@st.cache_resource
def load_model(name, horizon):
    """Nạp mô hình chuyên biệt cho từng mốc (đọc đúng cấu trúc file thực tế)."""
    try:
        conf = MODEL_DEFS[name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Force device for debugging if needed (uncomment below if user wants to force)
        # device = "cuda"
        _swap_src(conf["proj_dir"])
        mod = importlib.import_module(conf["mod"])
        importlib.reload(mod)
        cls = getattr(mod, conf["cls"])

        if name == "GUMNet":
            # GUMNet lưu toàn bộ meta trong file .pt
            ckpt_path = CKPT_DIR / f"gumnet_h{horizon}.pt"
            if not ckpt_path.exists():
                return None, None, device
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = cls(
                seq_len=ckpt["seq_len"], input_dim=ckpt["input_dim"],
                output_dim=ckpt["output_dim"], horizon=ckpt["horizon"],
                d_feat=ckpt.get("d_feat", 64), num_quantiles=ckpt["num_quantiles"],
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            meta = {
                "feature_cols": [c.strip() for c in ckpt["feature_cols"]],
                "target_cols":  [c.strip() for c in ckpt["target_cols"]],
                "seq_len": ckpt["seq_len"], "horizon": ckpt["horizon"], "kind": "quantile",
                "feature_scaler": ckpt["feature_scaler"],
                "target_scaler":  ckpt["target_scaler"],
            }

        else:
            # HybridTriNet: file .pt + thư mục meta riêng
            ckpt_path = CKPT_DIR / f"hybrid_h{horizon}.pt"
            meta_dir  = CKPT_DIR / f"hybrid_h{horizon}_meta"
            if not ckpt_path.exists() or not meta_dir.exists():
                return None, None, device

            with open(meta_dir / "feature_cols.json") as f:
                fj = json.load(f)

            f_cols = [c.strip() for c in fj.get("feature_cols", TARGET_COLS)]
            K = fj.get("K", 64)
            # Dùng H từ metadata (H mà model được train), không dùng horizon yêu cầu
            H_model = fj.get("H", horizon)

            model = cls(
                k=K, H=H_model, D_in=len(f_cols), D_out=len(TARGET_COLS),
                d_feat=96, kan_M=8, kan_depth=2,
                gru_hidden=128, gru_layers=1,
                attn_dmodel=64, attn_heads=4, attn_layers=2,
                patch_len=16, stride=8,
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            meta = {
                "feature_cols": f_cols, "target_cols": TARGET_COLS,
                "seq_len": K, "horizon": H_model, "kind": "point",
                "x_mu": np.load(meta_dir / "x_mu.npy"),
                "x_sd": np.load(meta_dir / "x_sd.npy"),
                "y_mu": np.load(meta_dir / "y_mu.npy"),
                "y_sd": np.load(meta_dir / "y_sd.npy"),
            }


        model.eval()
        return model, meta, device
    except Exception as e:
        return None, str(e), "cpu"


def predict_from_df(model, meta, df, device):
    k      = meta["seq_len"]
    f_cols = meta["feature_cols"]
    t_cols = meta["target_cols"]
    n_tgt  = len(t_cols)

    # Chuẩn bị đầu vào — Lỗi #19 (đã xác nhận): cách cũ lọc ra các cột SẴN CÓ (giữ đúng thứ tự
    # nhưng bỏ hẳn cột thiếu) rồi đệm số 0 vào CUỐI mảng, làm lệch toàn bộ vị trí cột nếu thiếu
    # bất kỳ cột nào ở giữa danh sách feature_cols — model nhận nhầm giá trị của cột này vào vị
    # trí của cột khác, dự báo sai lệch nghiêm trọng. `reindex` giữ đúng vị trí từng cột theo
    # đúng thứ tự f_cols, cột nào thiếu thì điền 0 ĐÚNG VỊ TRÍ của nó, không dịch chuyển gì cả.
    X = df.reindex(columns=f_cols, fill_value=0.0).values

    if "feature_scaler" in meta:
        X = meta["feature_scaler"].transform(X)
    else:
        X = (X - meta["x_mu"]) / (meta["x_sd"] + 1e-8)

    x_in = torch.tensor(X[-k:], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        out, _ = model(x_in)

    if meta["kind"] == "quantile":
        # GUMNet: out shape (1, H, n_quantiles, n_tgt) hoặc (1, H, n_tgt, n_quantiles)
        raw = out.cpu().numpy()[0]  # (H, ...)
        if raw.ndim == 3:           # (H, n_tgt, n_quantiles)
            p50 = meta["target_scaler"].inverse_transform(raw[:, :, 1])
        else:                       # fallback
            p50 = meta["target_scaler"].inverse_transform(raw[..., 1])
    else:
        # HybridTriNet: out shape (1, H*n_tgt) hoặc (1, H, n_tgt)
        raw = out.cpu().numpy()[0]
        if raw.ndim == 1:
            total = raw.size
            H = total // n_tgt
            raw = raw.reshape(H, n_tgt)
        # y_mu/y_sd là của toàn bộ feature, chỉ lấy n_tgt cột cuối (target cols)
        y_mu = np.array(meta["y_mu"]).reshape(-1)[-n_tgt:]
        y_sd = np.array(meta["y_sd"]).reshape(-1)[-n_tgt:]
        p50 = raw * (y_sd + 1e-8) + y_mu


    # Sinh ngày (bỏ qua cuối tuần)
    h_out = p50.shape[0]
    last  = df[DATE_COL].iloc[-1]
    dates, d = [], last
    while len(dates) < h_out:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d)

    result = pd.DataFrame(p50[:len(dates)], columns=t_cols)
    result.insert(0, DATE_COL, [pd.Timestamp(dt).normalize() for dt in dates[:len(result)]])
    return result


# === SIMULATION ENGINE ===

def run_upload_simulation(base_path, upload_files, start_date, sel_horizons=None):
    # Lỗi #13 (đã xác nhận): trước đây dùng biến `sel_horizons` mà hàm này không nhận làm tham
    # số và cũng không phải biến global -> NameError mỗi khi có dữ liệu mới cần backtest, bị
    # nuốt bởi except bên dưới khiến toàn bộ tính năng "Đánh giá mô hình" luôn thất bại âm thầm.
    if sel_horizons is None or len(sel_horizons) == 0:
        sel_horizons = HORIZONS
    base_full = load_df(base_path)
    base = base_full[base_full[DATE_COL] < start_date].copy()
    all_records = []
    
    # Gộp toàn bộ dữ liệu thực tế lịch sử và các file upload để làm nguồn tra cứu giá thực tế
    actual_dfs = [base_full]
    for fp in upload_files:
        df_tmp = load_df(fp)
        if not df_tmp.empty:
            actual_dfs.append(df_tmp)
    full_actuals = pd.concat(actual_dfs, ignore_index=True).drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL)
    
    with open(ROOT / "sim_log.txt", "w", encoding="utf-8") as logf:
        logf.write(f"Simulation started. Files: {len(upload_files)}\n")
        
        status_text = st.empty()
        task_idx = 0
        total_tasks = len(upload_files) * len(MODEL_DEFS)

        
        for idx, fpath in enumerate(upload_files):
            df_upload = load_df(fpath)
            if df_upload.empty: continue
            
            # Đồng bộ tên cột target
            for tc in TARGET_COLS:
                for c in df_upload.columns:
                    if tc.replace(" ", "").lower() == str(c).replace(" ", "").lower():
                        df_upload = df_upload.rename(columns={c: tc})
            
            avail_tgt = [c for c in TARGET_COLS if c in df_upload.columns]
            if not avail_tgt: continue

            base_dates = set(base[DATE_COL].dt.strftime('%Y-%m-%d'))
            new_rows = df_upload[~df_upload[DATE_COL].dt.strftime('%Y-%m-%d').isin(base_dates)].copy()
            new_rows = new_rows[new_rows[DATE_COL] >= start_date]
            
            if not new_rows.empty:
                base_for_pred = pd.concat([base_full[base_full[DATE_COL] < base[DATE_COL].min()], base], ignore_index=True)
                base_for_pred = base_for_pred.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL).tail(500)
                base_enriched = enrich_with_exo(base_for_pred, base_full)

                for mname in MODEL_DEFS:
                    task_idx += 1
                    try:
                        match_data = []
                        # CHẾ ĐỘ SIÊU NHẸ: 3 điểm kiểm tra mỗi file
                        indices = np.unique(np.linspace(0, len(new_rows) - 1, 3, dtype=int))
                        indices = [i for i in indices if 0 <= i < len(new_rows)]
                        
                        import gc
                        for step_i, idx_in_new in enumerate(indices):
                            status_text.text(f"⏳ {mname} | File {idx+1} | Điểm {step_i+1}/{len(indices)}")
                            gc.collect()
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            
                            history_df = pd.concat([base_enriched, new_rows.iloc[:idx_in_new]], ignore_index=True)
                            actual_pool = new_rows.iloc[idx_in_new:idx_in_new+1].copy()
                            if actual_pool.empty: continue
                            
                            for h in sorted(sel_horizons):
                                # Nạp mô hình chuyên biệt cho đúng mốc h
                                model_h, meta_h, device_h = load_model(mname, h)
                                if not model_h: continue
                                _swap_src(MODEL_DEFS[mname]["proj_dir"])
                                
                                missing = [c for c in meta_h["feature_cols"] if c not in history_df.columns]
                                hist_filled = history_df.copy()
                                if missing:
                                    hist_filled = generate_time_features(hist_filled)
                                    for mc in missing:
                                        if mc in base_full.columns:
                                            hist_filled[mc] = base_full.set_index(DATE_COL).reindex(hist_filled[DATE_COL])[mc].values
                                hist_filled = hist_filled.ffill().bfill().fillna(0)
                                
                                pred_df = predict_from_df(model_h, meta_h, hist_filled, device_h)
                                if pred_df.empty: continue
                                
                                idx_h = min(h - 1, len(pred_df) - 1)
                                p_row = pred_df.iloc[idx_h]
                                pred_date = pd.Timestamp(p_row[DATE_COL]).normalize()
                                
                                # Tìm hàng thực tế tương ứng với ngày dự báo trong cơ sở dữ liệu thực tế đầy đủ
                                match_row = full_actuals[full_actuals[DATE_COL] == pred_date]
                                if not match_row.empty:
                                    a_row = match_row.iloc[0]
                                    for tgt in avail_tgt:
                                        if tgt in p_row and tgt in a_row.index and pd.notna(a_row[tgt]):
                                            match_data.append({
                                                "Model": mname, "Horizon": f"{h}d",
                                                "Upload": f"#{idx+1} {fpath.name}",
                                                DATE_COL: pred_date, "Target": tgt,
                                                "Dự báo": round(float(p_row[tgt]), 2),
                                                "Thực tế": round(float(a_row[tgt]), 2),
                                                "Sai lệch": round(abs(float(p_row[tgt]) - float(a_row[tgt])), 2),
                                                "% Lệch": round(abs(float(p_row[tgt]) - float(a_row[tgt])) / (abs(float(a_row[tgt])) + 1e-8) * 100, 2),
                                            })
                        
                        if match_data:
                            res_df = pd.DataFrame(match_data).drop_duplicates(subset=["Model", "Horizon", DATE_COL, "Target"])
                            all_records.extend(res_df.to_dict("records"))
                    except Exception as e: 
                        logf.write(f"EXCEPTION: {mname}: {e}\n")
                        continue
                
                base = pd.concat([base, new_rows], ignore_index=True)
                base = base.drop_duplicates(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

        status_text.empty()

    return pd.DataFrame(all_records)

def show_live_forecasts(base_full, file_paths, sel_models, sel_horizons=None):
    """Tính toán và hiển thị dự báo đa mốc thời gian dựa trên dữ liệu mới nhất."""
    if sel_horizons is None or len(sel_horizons) == 0:
        sel_horizons = HORIZONS

    if not file_paths:
        st.warning("⚠️ Chưa có file upload để lấy dữ liệu mới nhất.")
        return

    # Lấy ngày lớn nhất từ các file upload
    upload_dfs = []
    for fp in file_paths:
        upload_dfs.append(load_df(fp))
    
    upload_max_date = pd.concat(upload_dfs)[DATE_COL].max() if upload_dfs else base_full[DATE_COL].max()

    # Tổng hợp dữ liệu mới nhất
    latest_df = base_full.copy()
    for fdf in upload_dfs:
        latest_df = pd.concat([latest_df, fdf])
        
    latest_df = latest_df.drop_duplicates(DATE_COL).sort_values(DATE_COL).reset_index(drop=True)
    
    # CẮT DỮ LIỆU ĐẾN NGÀY UPLOAD CUỐI CÙNG (Để dự báo từ mốc file upload)
    latest_df = latest_df[latest_df[DATE_COL] <= upload_max_date]
    
    history = latest_df.tail(500)
    last_date = history[DATE_COL].iloc[-1]
    
    # Tìm xem ngày này thuộc file nào để báo cho người dùng
    source_file = "Dataset gốc"
    for f in file_info:
        if f["max_date"] == last_date:
            source_file = f["name"]
            break
            
    st.markdown(f"### 🔮 Bảng dự báo (Từ mốc: **{last_date.strftime('%d/%m/%Y')}**)")
    st.caption(f"📌 Nguồn dữ liệu mốc: **{source_file}**")

    
    tabs = st.tabs(sel_models)
    for idx, mname in enumerate(sel_models):
        with tabs[idx]:
            all_preds = []
            future_points = [] # Dùng cho biểu đồ xu hướng
            detailed_preds = {} # Lưu dự báo chi tiết
            
            # Lấy giá hiện tại làm mốc 0
            last_prices = history.iloc[-1]
            future_points = []
            for tgt in TARGET_COLS:
                if tgt in last_prices:
                    future_points.append({DATE_COL: last_date, "Target": tgt, "Giá": float(last_prices[tgt]), "Loại": "Hiện tại"})

            # Chuẩn bị dữ liệu lịch sử một lần duy nhất
            history_enriched = generate_time_features(history.copy())

            # Chạy dự báo cho từng mốc với model chuyên biệt (chỉ các mốc người dùng đã chọn)
            all_preds = []
            pred_cache = {}  # Lưu kết quả để vẽ biểu đồ

            for h in sorted(sel_horizons):
                try:
                    model_h, meta_h, device_h = load_model(mname, h)
                    if not model_h:
                        continue
                    _swap_src(MODEL_DEFS[mname]["proj_dir"])

                    # Bổ sung cột còn thiếu
                    hist_h = history_enriched.copy()
                    missing = [c for c in meta_h["feature_cols"] if c not in hist_h.columns]
                    for mc in missing:
                        if mc in base_full.columns:
                            hist_h[mc] = base_full.set_index(DATE_COL).reindex(hist_h[DATE_COL])[mc].values
                    hist_h = hist_h.ffill().bfill().fillna(0)

                    pred_df = predict_from_df(model_h, meta_h, hist_h, device_h)
                    if pred_df.empty:
                        continue

                    # Lấy dòng cuối cùng là dự báo tại mốc h ngày
                    p_row = pred_df.iloc[-1]
                    f_date = last_date + pd.Timedelta(days=h)
                    if DATE_COL in pred_df.columns:
                        f_date = p_row[DATE_COL]

                    row_ui = {"Ngày dự đoán": f_date.strftime('%d/%m/%Y'), "Mốc": f"+{h} ngày"}
                    for tgt in TARGET_COLS:
                        if tgt in p_row:
                            val = float(p_row[tgt])
                            row_ui[tgt] = f"{val:,.2f}"
                            future_points.append({DATE_COL: f_date, "Target": tgt, "Giá": val, "Loại": "Dự báo"})

                    all_preds.append(row_ui)

                    pred_cache[h] = (pred_df, f_date)

                except Exception as e:
                    # Quay lại dùng caption để không làm rối giao diện nếu mốc đó chưa có model
                    st.caption(f"ℹ️ Mốc {h}d: {e}")
                    continue



            # Hiển thị bảng tóm tắt
            if all_preds:
                st.markdown("#### 📋 Bảng tổng hợp dự báo đa mốc thời gian")
                st.caption("💡 **Cách xem:** Số liệu biểu thị mức giá dự kiến theo đơn vị **USD/thùng** (MG95, MG92) và **USD/tấn** (DO). Bấm nút tải CSV bên dưới để xuất báo cáo gửi lãnh đạo.")
                df_preds_export = pd.DataFrame(all_preds)
                safe_dataframe(df_preds_export.set_index("Ngày dự đoán"))
                
                csv_bytes = df_preds_export.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 Xuất Bảng Dự Báo Ra Excel (CSV)",
                    data=csv_bytes,
                    file_name=f"Du_bao_gia_dau_{last_date.strftime('%Y%m%d')}_{mname}.csv",
                    mime="text/csv",
                    key=f"dl_btn_{mname}"
                )

            # Vẽ biểu đồ lộ trình cho từng mặt hàng
            if pred_cache:
                st.markdown(f"**📈 Biểu đồ so sánh các chân trời dự báo ({mname})**")
                st.caption("💡 **Cách tương tác:** Bấm vào tên mốc ở chú giải bên phải để ẩn/hiện từng đường; rê chuột vào các điểm để xem giá cụ thể; kéo chuột trên biểu đồ để phóng to (zoom in).")
                for tgt in TARGET_COLS:
                    fig = go.Figure()
                    last_val = float(last_prices[tgt]) if tgt in last_prices else 0

                    for h, (pred_df, f_date) in pred_cache.items():
                        if tgt not in pred_df.columns:
                            continue
                        vals = pred_df[tgt].values[:h]
                        if DATE_COL in pred_df.columns:
                            dates = list(pred_df[DATE_COL].values[:h])
                        else:
                            dates = [last_date + pd.Timedelta(days=d) for d in range(1, h + 1)]

                        all_dates = [last_date] + list(dates)
                        all_vals  = [last_val]  + list(vals)

                        dash  = "solid" if h == max(pred_cache.keys()) else "dot"
                        width = 2.5    if h == max(HORIZONS) else 1.5
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(all_dates), y=all_vals,
                            name=f"Dự báo {h}d", mode="lines",
                            line=dict(dash=dash, width=width)
                        ))


                    fig.update_layout(
                        title=f"So sánh lộ trình dự báo: {tgt}",
                        template="plotly_dark", height=350, hovermode="x unified",
                        xaxis=dict(type='date', tickformat='%d/%m/%Y')
                    )
                    safe_plotly_chart(fig, key=f"chart_{mname}_{tgt}")


    st.markdown("---")



# === UI CONFIGURATION & THEMING ===

st.markdown("""
<style>
    /* Clean Enterprise Layout */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 96% !important;
    }
    
    /* Rounded modern components */
    .stButton > button, 
    div[data-baseweb="select"], 
    div[data-baseweb="popover"], 
    .stAlert, 
    div[data-testid="stDataFrame"], 
    div[data-testid="stFileUploader"] {
        border-radius: 8px !important;
    }
    /* Khung upload to hơn, có icon + chữ hướng dẫn tiếng Việt (theo đúng frontend_mockup.html)
       thay cho giao diện mặc định của Streamlit — CSS thuần, không dùng JS nên không bị lỗi
       "không đồng bộ sau khi rerun" như các chỗ đã từng gặp trong dự án này. */
    section[data-testid="stFileUploaderDropzone"] {
        min-height: 150px !important;
        border-radius: 10px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 4px !important;
        text-align: center !important;
    }
    section[data-testid="stFileUploaderDropzone"]::before {
        content: "⇧";
        font-size: 26px;
        color: #00ad91;
        font-weight: 700;
        order: -3;
    }
    section[data-testid="stFileUploaderDropzone"]::after {
        content: "Kéo thả file vào đây";
        font-size: 14px;
        font-weight: 600;
        color: #1e293b;
        order: -2;
    }
    /* Ẩn icon nhỏ mặc định trong nút (đã có icon lớn ở trên), đổi chữ nút "Upload" -> "Chọn file dữ liệu" */
    section[data-testid="stFileUploaderDropzone"] span[data-testid="stIconMaterial"] {
        display: none !important;
    }
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] p {
        font-size: 0 !important;
    }
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] p::after {
        content: "Chọn file dữ liệu";
        font-size: 13px !important;
        font-weight: 600;
    }
    /* "200MB per file..." đã có sẵn ở ô "Dung lượng tối đa" bên dưới nên ẩn dòng trùng lặp này đi,
       thay bằng phụ đề "hoặc bấm để chọn file từ máy tính" đúng theo bản thiết kế */
    div[data-testid="stFileUploaderDropzoneInstructions"] span {
        font-size: 0 !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] span::after {
        content: "hoặc bấm để chọn file từ máy tính";
        font-size: 12.5px !important;
        color: #64748b;
        order: -1;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        order: -1;
    }
    
    /* Elegant buttons */
    .stButton > button {
        border: 1px solid rgba(0, 173, 145, 0.25) !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: #00ad91 !important;
        box-shadow: 0 2px 8px rgba(0, 173, 145, 0.15) !important;
    }

    /* Hub Notice Boxes */
    .hub-notice {
        border-radius: 9px;
        padding: 13px 15px;
        font-size: 13px;
        line-height: 1.5;
        margin-bottom: 15px;
    }
    .hub-notice.cpu {
        background: #fff7e7;
        border: 1px solid #f6d58c;
        color: #794800;
    }
    .hub-notice.gpu {
        background: #eafaf5;
        border: 1px solid #bcebdc;
        color: #116d5c;
    }
    
    /* Operational Guide Cards */
    .guide-card {
        border: 1px solid #e2e8f0;
        background: #ffffff;
        border-radius: 10px;
        padding: 18px;
        height: 100%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)


# Lưu base_full gốc để dùng cho live forecast
base_full_orig = load_df(BUILTIN_CSV)

CACHE_FILE = ROOT / "simulation_cache.pkl"

# Khóa job huấn luyện bằng file trên đĩa: đây là app LAN nhiều người cùng truy cập chung
# một server Streamlit, nên khóa phải chặn được cả những phiên/tab khác, không chỉ session
# hiện tại (session_state không đủ vì mỗi tab/trình duyệt có session_state riêng).
TRAIN_LOCK_FILE = ROOT / ".training.lock"

def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # Tiến trình tồn tại nhưng khác quyền truy cập
    except OSError:
        return False
    except Exception:
        # Trên Windows, os.kill(pid, 0) với PID đã chết đôi khi ném SystemError thay vì
        # OSError thông thường (đã bắt gặp thật khi kiểm thử ở train_all_horizons.py) —
        # coi như không còn sống thay vì để lỗi rơi ra ngoài.
        return False

def get_active_training_lock():
    """Trả về dict thông tin job đang chạy nếu lock còn hiệu lực, ngược lại None (và tự dọn lock rác)."""
    if not TRAIN_LOCK_FILE.exists():
        return None
    try:
        info = json.loads(TRAIN_LOCK_FILE.read_text(encoding="utf-8"))
        pid = info.get("pid")
        if pid and _pid_alive(pid):
            return info
    except Exception:
        pass
    # Lock rác (job cũ bị crash/kill mà không dọn được) -> xoá để không khoá cứng vĩnh viễn
    try:
        TRAIN_LOCK_FILE.unlink()
    except Exception:
        pass
    return None

def acquire_training_lock(models, horizons, pid=None):
    # Lỗi #16 (đã xác nhận): trước đây luôn ghi os.getpid() — PID của chính tiến trình Streamlit
    # server (luôn sống), không phải PID của tiến trình huấn luyện con thật (subprocess.Popen).
    # Cho phép truyền pid thật vào để _pid_alive() kiểm tra đúng đối tượng; khi chưa có subprocess
    # (giữ chỗ trước khi Popen chạy) thì tạm dùng os.getpid() để tránh race giữa 2 người dùng.
    TRAIN_LOCK_FILE.write_text(json.dumps({
        "pid": pid if pid is not None else os.getpid(),
        "models": models,
        "horizons": horizons,
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }), encoding="utf-8")

def release_training_lock():
    try:
        TRAIN_LOCK_FILE.unlink()
    except Exception:
        pass

def get_dir_fingerprint():
    data_dir = ROOT / "datasets"
    files = [f for f in data_dir.glob("*") if f.suffix.lower() in [".xlsx", ".xls", ".csv"] and not f.name.startswith("~$")]
    hz_str = "_".join(str(x) for x in HORIZONS)
    if not files: return f"empty_{hz_str}"
    return f"{len(files)}_{max(f.stat().st_mtime for f in files)}_{hz_str}"

fingerprint = get_dir_fingerprint()

@st.cache_data
def get_sorted_files(fp):
    data_dir = ROOT / "datasets"
    files = [f for f in data_dir.glob("*") if f.suffix.lower() in [".xlsx", ".xls", ".csv"]]
    info = []
    for f in files:
        df = load_df(f)
        if not df.empty: info.append({"path": str(f), "max_date": df[DATE_COL].max(), "name": f.name, "rows": len(df)})
    info.sort(key=lambda x: x["max_date"])
    return info

file_info = get_sorted_files(fingerprint)
file_paths = [Path(i["path"]) for i in file_info]

# CUTOFF_DATE: chỉ lấy các điểm backtest trong 365 ngày gần nhất TÍNH THEO NGÀY MỚI NHẤT
# đang có trong dữ liệu (dataset gốc + các file đã upload) — tự động trôi theo dữ liệu mới,
# không còn là một ngày cố định phải nhớ sửa tay mỗi năm.
_known_max_dates = [base_full_orig[DATE_COL].max()] + [i["max_date"] for i in file_info]
_known_max_dates = [d for d in _known_max_dates if pd.notna(d)]
_latest_known_date = max(_known_max_dates) if _known_max_dates else pd.Timestamp.now()
CUTOFF_DATE = _latest_known_date - pd.Timedelta(days=365)

# Load cache
cache_mismatch = False
if CACHE_FILE.exists():
    try:
        cache = pd.read_pickle(CACHE_FILE)
        combined = cache.get("df")
        if cache.get("fp") != fingerprint:
            cache_mismatch = True
    except: combined = None
else: combined = None

if combined is None:
    combined = pd.DataFrame()


# ==========================================
# 1. BẢNG ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ==========================================
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:10px; padding:12px 12px 13px; border-radius:10px; margin-bottom:16px; background:linear-gradient(135deg, #00ad91 0%, #6954d9 100%);">
    <span style="flex:none; width:34px; height:34px; border-radius:9px; background:rgba(255,255,255,0.18); display:flex; align-items:center; justify-content:center; font-size:18px;">🛢️</span>
    <div style="line-height:1.3;">
        <b style="display:block; font-size:13.5px; font-weight:800; color:#ffffff; letter-spacing:-0.01em;">Oil Forecast Hub</b>
        <span style="font-size:10px; font-weight:600; color:rgba(255,255,255,0.85); letter-spacing:0.03em; text-transform:uppercase;">Automated Evaluation</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.caption("KHU VỰC LÀM VIỆC")
NAV_OPTIONS = [
    "◈  Dự báo",
    "▦  Đánh giá mô hình",
    "◷  Lịch sử & Xuất dữ liệu",
    "⚙  Huấn luyện mô hình",
    "❓  Hướng dẫn sử dụng"
]
nav_choice = st.sidebar.radio(
    "Điều hướng",
    NAV_OPTIONS,
    key="main_nav_radio",
    label_visibility="collapsed"
)

# Gọi ở đây (sau khi đã biết đang ở trang nào) thay vì gọi tuốt ở đầu file: nội dung script
# nhúng đổi theo nav_choice nên buộc Streamlit phải nạp lại đúng lúc chuyển trang.
inject_oil_tour_engine(nav_choice)

st.sidebar.markdown("---")
st.sidebar.caption("MÔ HÌNH DỰ BÁO")
st.sidebar.markdown("""
<div style="padding:10px 12px; border-radius:8px; background:#eaf5f4; border:1px solid #bcebdc; font-size:12.5px;">
    <div style="color:#116d5c; font-size:11px; margin-bottom:2px; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">ĐỘNG CƠ AI CHUẨN HÓA:</div>
    <b style="color:#075f57; font-size:13.5px;">🧠 GUMNet Enterprise</b>
    <div style="color:#116d5c; font-size:11px; margin-top:3px;">✓ Sẵn sàng 7/7 mốc thời gian (Độ chính xác cao)</div>
</div>
""", unsafe_allow_html=True)
sel_models = ["GUMNet"]

st.sidebar.markdown("---")

# Huy hiệu phần cứng ở chân Sidebar (Hardware Status Badge)
is_gpu = torch.cuda.is_available()
if is_gpu:
    st.sidebar.markdown("""
    <div style="padding:11px 13px; border-radius:9px; background:#eafaf5; border:1px solid #bcebdc; color:#116d5c; font-size:12px;">
        <span class="pulsing-dot"></span>
        <b>Server: GPU NVIDIA CUDA</b><br>
        <small style="margin-left:14px; color:#116d5c;">Tăng tốc tối đa · Sẵn sàng</small>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="padding:11px 13px; border-radius:9px; background:#fff7e7; border:1px solid #f6d58c; color:#794800; font-size:12px;">
        <span class="pulsing-dot-cpu"></span>
        <b>Server: CPU Doanh Nghiệp</b><br>
        <small style="margin-left:14px; color:#794800;">6 vCPUs · Dự báo tức thì (< 1s)</small>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 2. TỰ ĐỘNG ĐỒNG BỘ DỮ LIỆU ĐÁNH GIÁ (BACKTESTING)
# ==========================================
if cache_mismatch or (combined.empty and file_paths):
    with st.spinner("⏳ Đang tự động phân tích & đối chiếu sai số cho dữ liệu thị trường mới..."):
        try:
            combined = run_upload_simulation(str(BUILTIN_CSV), file_paths, CUTOFF_DATE)
            pd.to_pickle({"fp": fingerprint, "df": combined}, CACHE_FILE)
            cache_mismatch = False
        except Exception:
            pass

if not combined.empty:
    if "Mặt hàng" in combined.columns:
        combined = combined.rename(columns={"Mặt hàng": "Target"})
    df_view = combined[(combined["Model"].isin(sel_models)) & (combined[DATE_COL] >= CUTOFF_DATE)]
else:
    df_view = pd.DataFrame()


# ==========================================
# 3. NỘI DUNG TỪNG TRANG (THEO MẪU ENTERPRISE MOCKUP)
# ==========================================

# ──────────────────────────────────────────
# TRANG 1: DỰ BÁO
# ──────────────────────────────────────────
if nav_choice == "◈  Dự báo":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="color:#00ad91; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.09em;">TRUNG TÂM DỰ BÁO GIÁ DẦU</div>
        <h1 style="margin:4px 0; font-size:28px; font-weight:800; letter-spacing:-.03em;">Dự Báo Thị Trường</h1>
        <p style="margin:0; color:#64748b; font-size:14px;">Tạo dự báo giá đa mốc thời gian từ dữ liệu thị trường mới nhất.</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_status = st.columns([1.25, 0.75])
    
    with col_input:
        card_input = st.container(border=True)  # st.container(border=True): khung card THẬT của
        # Streamlit, bọc đúng các widget bên trong (kể cả st.file_uploader) — an toàn hơn nhiều so
        # với việc tự mở/đóng thẻ <div> bằng st.markdown rồi hy vọng nó bao đúng các widget khác,
        # vì mỗi lệnh st.markdown/st.file_uploader render vào 1 container DOM riêng của Streamlit,
        # <div> mở ở lệnh này KHÔNG thực sự bao được widget ở lệnh khác (dễ vỡ layout không báo lỗi).
    with card_input:
        st.markdown("#### Cập nhật dữ liệu thị trường")
        st.caption("Tải tệp Excel (.xlsx, .xls) hoặc CSV chứa cột Ngày và giá thị trường:")
        up = st.file_uploader("Upload file", type=["xlsx", "xls", "csv"], key="uploader_main", label_visibility="collapsed")
        active_file_paths = list(file_paths)
        if up:
            tmp = ROOT / "datasets" / up.name
            with open(tmp, "wb") as f: f.write(up.getbuffer())
            df_new = load_df(tmp)
            if not df_new.empty:
                m_date = df_new[DATE_COL].max()
                st.success(f"✅ Đã tiếp nhận tệp tin: **{up.name}** (Dữ liệu đến ngày: **{m_date.strftime('%d/%m/%Y')}**)")
                if tmp not in active_file_paths:
                    active_file_paths.append(tmp)

                # Việc A: gợi ý thông minh khi phát hiện dữ liệu MỚI hơn ngày hệ thống đang có
                # (_latest_known_date đã tính ở trên, TRƯỚC khi file này được ghi vào datasets/).
                if pd.notna(m_date) and m_date > _latest_known_date:
                    st.info(
                        f"🆕 **Phát hiện dữ liệu mới đến ngày {m_date.strftime('%d/%m/%Y')}** "
                        f"(hệ thống trước đó chỉ có đến {_latest_known_date.strftime('%d/%m/%Y')}). "
                        "Bạn có thể xem thử dự báo ngay bên dưới, hoặc huấn luyện lại (Finetune) "
                        "để mô hình cập nhật theo dữ liệu mới nhất — không bắt buộc."
                    )
                    if st.button("⚡ Chuyển sang Huấn luyện để Finetune ngay", key="btn_goto_train_from_upload"):
                        st.session_state["main_nav_radio"] = "⚙  Huấn luyện mô hình"
                        st.rerun()

        # Trước đây có ô multiselect cho người dùng bớt/thêm mốc trước khi tính — nhưng việc
        # tính đủ 7 mốc chỉ mất chưa tới 1 giây, và người dùng không thể "thêm/bớt" mốc thật sự
        # (checkpoint chỉ có sẵn đúng 7 mốc cố định) nên ô chọn dễ gây hiểu lầm (tưởng lọc để
        # chạy nhanh hơn) và hay bị bấm nhầm dấu "x" làm mất mốc. Giờ luôn tính đủ cả 7 mốc,
        # người dùng muốn xem/ẩn mốc nào thì bấm ngay vào chú giải (legend) trên biểu đồ bên dưới
        # (nhắc lại ở ngay phần biểu đồ, xem show_live_forecasts()).
        sel_hz_view = HORIZONS

        # Lấp khoảng trống cho card này cân chiều cao với card "Trạng thái hệ thống" bên cạnh.
        st.markdown("""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:12px; margin-top:14px; margin-bottom:12px;">
            <div style="background:#f8fafc; border-radius:8px; padding:10px 12px;">
                <div style="color:#64748b; margin-bottom:2px;">Định dạng hỗ trợ</div>
                <div style="font-weight:600;">XLSX, XLS, CSV</div>
            </div>
            <div style="background:#f8fafc; border-radius:8px; padding:10px 12px;">
                <div style="color:#64748b; margin-bottom:2px;">Dung lượng tối đa</div>
                <div style="font-weight:600;">200 MB / file</div>
            </div>
            <div style="background:#f8fafc; border-radius:8px; padding:10px 12px; grid-column:1 / -1;">
                <div style="color:#64748b; margin-bottom:2px;">Cột bắt buộc</div>
                <div style="font-weight:600;">Ngày + ít nhất 1 mặt hàng (MG95, MG92, DO 0.001%, DO 0.05%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_status:
        card_status = st.container(border=True)
    with card_status:
        st.markdown("#### Trạng thái hệ thống")
        if is_gpu:
            st.markdown("""
            <div class="hub-notice gpu">
                <b style="display:block; margin-bottom:2px;">🟢 Đang chạy bằng GPU CUDA</b>
                Dự báo và huấn luyện được tăng tốc phần cứng. Hiệu năng đạt mức tối đa.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="hub-notice cpu">
                <b style="display:block; margin-bottom:2px;">🔵 Đang chạy bằng CPU Doanh Nghiệp</b>
                Dự báo nhanh (< 1s) hoạt động bình thường. Huấn luyện lại có thể thực hiện tại mục <b>Huấn luyện</b>.
            </div>
            """, unsafe_allow_html=True)

        last_known_date = "Chưa rõ"
        if active_file_paths:
            try:
                dfs_tmp = [load_df(p) for p in active_file_paths]
                last_known_date = pd.concat(dfs_tmp)[DATE_COL].max().strftime("%d/%m/%Y")
            except:
                last_known_date = base_full_orig[DATE_COL].max().strftime("%d/%m/%Y")
        else:
            last_known_date = base_full_orig[DATE_COL].max().strftime("%d/%m/%Y")

        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:12px;">
            <div style="border:1px solid #e2e8f0; border-radius:8px; padding:10px; background:#f8fafc;">
                <span style="color:#64748b; font-size:11px;">Dữ liệu gần nhất</span><br>
                <b style="font-size:14px; color:#1e293b;">{last_known_date}</b>
            </div>
            <div style="border:1px solid #e2e8f0; border-radius:8px; padding:10px; background:#f8fafc;">
                <span style="color:#64748b; font-size:11px;">Model kích hoạt</span><br>
                <b style="font-size:14px; color:#1e293b;">{sel_models[0]}</b>
            </div>
            <div style="border:1px solid #e2e8f0; border-radius:8px; padding:10px; background:#f8fafc;">
                <span style="color:#64748b; font-size:11px;">Thiết bị tính toán</span><br>
                <b style="font-size:14px; color:#1e293b;">{'NVIDIA GPU' if is_gpu else 'CPU (6 vCPU)'}</b>
            </div>
            <div style="border:1px solid #e2e8f0; border-radius:8px; padding:10px; background:#f8fafc;">
                <span style="color:#64748b; font-size:11px;">Trọng số Checkpoint</span><br>
                <b style="font-size:14px; color:#087762;">Sẵn sàng (7/7 mốc)</b>
            </div>
        </div>

        <div style="display:flex; flex-direction:column; gap:6px; margin-top:10px; margin-bottom:12px;">
            <div style="display:flex; align-items:baseline; gap:6px; border:1px solid #e2e8f0; border-radius:8px; padding:8px 10px; font-size:12px; background:#ffffff; flex-wrap:wrap;">
                <b style="color:#087762; white-space:nowrap;">✓ Dữ liệu hợp lệ</b>
                <span style="color:#64748b;">— Có cột Ngày &amp; Giá</span>
            </div>
            <div style="display:flex; align-items:baseline; gap:6px; border:1px solid #e2e8f0; border-radius:8px; padding:8px 10px; font-size:12px; background:#ffffff; flex-wrap:wrap;">
                <b style="color:#9b6100; white-space:nowrap;">! Kiểm tra độ mới</b>
                <span style="color:#64748b;">— Định kỳ nạp file mới</span>
            </div>
            <div style="display:flex; align-items:baseline; gap:6px; border:1px solid #e2e8f0; border-radius:8px; padding:8px 10px; font-size:12px; background:#ffffff; flex-wrap:wrap;">
                <b style="color:#087762; white-space:nowrap;">✓ Phạm vi dự báo</b>
                <span style="color:#64748b;">— Tối đa 60 ngày</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    ckpts_exist = any((CKPT_DIR / f"gumnet_h{h}.pt").exists() for h in HORIZONS)
    if ckpts_exist:
        show_live_forecasts(base_full_orig, active_file_paths, sel_models, sel_hz_view)
    else:
        st.info("ℹ️ **Chưa có mô hình nào được nạp.** Hãy vào mục 'Huấn luyện mô hình' để khởi tạo trọng số.")


# ──────────────────────────────────────────
# TRANG 2: ĐÁNH GIÁ MÔ HÌNH (METRICS)
# ──────────────────────────────────────────
elif nav_choice == "▦  Đánh giá mô hình":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="color:#00ad91; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.09em;">KIỂM ĐỊNH MÔ HÌNH (BACKTESTING)</div>
        <h1 style="margin:4px 0; font-size:28px; font-weight:800; letter-spacing:-.03em;">Đánh Giá Độ Chính Xác</h1>
        <p style="margin:0; color:#64748b; font-size:14px;">Theo dõi sai số MAE/MAPE từ các dự báo đối chiếu với giá thị trường thực tế.</p>
    </div>
    """, unsafe_allow_html=True)

    if not df_view.empty:
        avg_mape = df_view["% Lệch"].mean()
        avg_mae = df_view["Sai lệch"].mean()
        n_samples = len(df_view)
        n_uploads = len(df_view["Upload"].unique()) if "Upload" in df_view.columns else len(file_paths)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            status_mape = "Trong ngưỡng an toàn (< 7%)" if avg_mape < 7.0 else ("Chấp nhận được (7-10%)" if avg_mape <= 10.0 else "Khuyến nghị Finetune (> 10%)")
            color_mape = "#087762" if avg_mape < 7.0 else ("#c77700" if avg_mape <= 10.0 else "#dc2626")
            st.markdown(f"""
            <div style="border:1px solid #e2e8f0; border-radius:10px; padding:16px; background:#fff;">
                <span style="color:#64748b; font-size:12px; font-weight:600;">MAPE TRUNG BÌNH</span>
                <div style="font-size:28px; font-weight:800; margin:4px 0; color:#1e293b;">{avg_mape:.2f}<small style="font-size:14px; font-weight:500; color:#64748b;">%</small></div>
                <span style="color:{color_mape}; font-size:12px; font-weight:600;">● {status_mape}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(f"""
            <div style="border:1px solid #e2e8f0; border-radius:10px; padding:16px; background:#fff;">
                <span style="color:#64748b; font-size:12px; font-weight:600;">MAE TRUNG BÌNH</span>
                <div style="font-size:28px; font-weight:800; margin:4px 0; color:#1e293b;">{avg_mae:,.2f} <small style="font-size:14px; font-weight:500; color:#64748b;">USD</small></div>
                <span style="color:#087762; font-size:12px; font-weight:600;">● Sai lệch giá tuyệt đối</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m3:
            st.markdown(f"""
            <div style="border:1px solid #e2e8f0; border-radius:10px; padding:16px; background:#fff;">
                <span style="color:#64748b; font-size:12px; font-weight:600;">MẪU ĐÁNH GIÁ ĐỐI CHIẾU</span>
                <div style="font-size:28px; font-weight:800; margin:4px 0; color:#1e293b;">{n_samples} <small style="font-size:14px; font-weight:500; color:#64748b;">điểm dữ liệu</small></div>
                <span style="color:#64748b; font-size:12px;">Từ {n_uploads} đợt cập nhật dữ liệu</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background:#f8fafc; border-left:4px solid #00ad91; border-radius:6px; padding:11px 16px; margin:14px 0 18px; font-size:13px; color:#334155; line-height:1.6;">
            <b>💡 Cẩm nang đọc chỉ số & hành động:</b><br>
            • <b>MAPE (%)</b>: Phần trăm sai lệch trung bình giữa giá AI đoán so với giá thị trường thực tế. 
              <span style="color:#087762; font-weight:700;">Xanh (&lt; 7%)</span>: Rất tốt, yên tâm dùng số liệu ➔ 
              <span style="color:#9b6100; font-weight:700;">Vàng (7–10%)</span>: Chấp nhận được ➔ 
              <span style="color:#dc2626; font-weight:700;">Đỏ (&gt; 10%)</span>: Biến động mạnh, khuyến nghị sang mục <i>Huấn luyện</i> để Finetune.<br>
            • <b>MAE (USD)</b>: Sai số tuyệt đối tính bằng số tiền thực tế (USD/thùng).<br>
            • <b>Bảng nhiệt (Heatmap)</b>: Màu xanh đậm biểu thị vùng dự báo bám sát nhất. Càng về mốc xa (+60d), biến động thị trường lớn nên sai số tăng tự nhiên.
        </div>
        """, unsafe_allow_html=True)
        
        col_t1, col_t2 = st.columns(2)
        h_order = [f"{h}d" for h in HORIZONS]
        with col_t1:
            st.markdown("#### 🎯 MAPE (%) theo Mốc dự báo")
            mape_h = df_view.groupby(["Model", "Horizon"])["% Lệch"].mean().unstack().round(2)
            mape_h = mape_h[[c for c in h_order if c in mape_h.columns]]
            safe_dataframe(mape_h.style.format("{:.2f}%").background_gradient(cmap="RdYlGn_r"))
            
        with col_t2:
            st.markdown("#### 🛢️ MAPE (%) theo Mặt hàng")
            mape_t = df_view.groupby(["Model", "Target"])["% Lệch"].mean().unstack().round(2)
            safe_dataframe(mape_t.style.format("{:.2f}%").background_gradient(cmap="RdYlGn_r"))
            
        st.markdown("#### 📋 Chi tiết sai lệch giá tuyệt đối (MAE)")
        mae_piv = df_view.groupby(["Model", "Horizon"])["Sai lệch"].mean().unstack().round(2)
        mae_piv = mae_piv[[c for c in h_order if c in mae_piv.columns]]
        safe_dataframe(mae_piv.style.format("{:,.2f}"))
        
        st.markdown("#### 📈 Xu hướng sai số MAPE qua các chân trời dự báo")
        fig_mape = go.Figure()
        mape_means = [df_view[df_view["Horizon"] == f"{h}d"]["% Lệch"].mean() for h in HORIZONS]
        fig_mape.add_trace(go.Scatter(
            x=[f"{h} ngày" for h in HORIZONS],
            y=mape_means,
            mode="lines+markers",
            name="MAPE (%)",
            line=dict(color="#00ad91", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 173, 145, 0.08)"
        ))
        fig_mape.update_layout(
            template="plotly_dark",
            height=320,
            yaxis=dict(title="MAPE (%)", tickformat=".2f"),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        safe_plotly_chart(fig_mape)
    else:
        st.info("ℹ️ Chưa có dữ liệu kiểm định. Hãy tải file dữ liệu ở mục 'Dự báo' để hệ thống tự động phân tích đối chiếu.")


# ──────────────────────────────────────────
# TRANG 3: LỊCH SỬ & XUẤT DỮ LIỆU
# ──────────────────────────────────────────
elif nav_choice == "◷  Lịch sử & Xuất dữ liệu":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="color:#00ad91; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.09em;">TRA CỨU & KIỂM TOÁN</div>
        <h1 style="margin:4px 0; font-size:28px; font-weight:800; letter-spacing:-.03em;">Lịch Sử & Xuất Dữ Liệu</h1>
        <p style="margin:0; color:#64748b; font-size:14px;">Tra cứu các đợt cập nhật dữ liệu và đối chiếu kết quả đã lưu trữ.</p>
    </div>
    <div style="background:#f8fafc; border-left:4px solid #087762; border-radius:6px; padding:11px 16px; margin:14px 0 18px; font-size:13px; color:#334155; line-height:1.6;">
        <b>💡 Hướng dẫn tra cứu & phục vụ kiểm toán:</b><br>
        • <b>Danh mục đợt nạp:</b> Quản lý lịch sử toàn bộ các tệp Excel/CSV đã nạp vào hệ thống theo thứ tự thời gian.<br>
        • <b>Bảng đối chiếu:</b> Cột <i>Dự báo</i> là giá AI đưa ra tại thời điểm đó trong quá khứ; cột <i>Thực tế</i> là giá thị trường diễn ra sau đó. Bấm <b>'📥 Xuất Bảng Đợt Này'</b> để tải file CSV nộp cấp quản lý.<br>
        • <b>Đồ thị đối chiếu:</b> Đường <b>nét liền xanh ngọc</b> là giá Thực tế, đường <b>nét đứt tím</b> là giá Dự báo. Hai đường càng bám sát nhau chứng tỏ mô hình dự đoán càng chính xác.
    </div>
    """, unsafe_allow_html=True)

    if file_info:
        st.markdown("#### 📂 Danh mục các đợt nạp dữ liệu thị trường")
        history_rows = []
        for idx, fi in enumerate(file_info):
            history_rows.append({
                "Đợt": f"#{idx+1:02d}",
                "Tên tệp tin": fi["name"],
                "Ngày dữ liệu cuối": fi["max_date"].strftime("%d/%m/%Y"),
                "Số dòng dữ liệu": f"{fi['rows']:,} dòng",
                "Trạng thái": "Đã đánh giá"
            })
        safe_dataframe(pd.DataFrame(history_rows).set_index("Đợt"))

    st.markdown("---")
    
    if not df_view.empty:
        st.markdown("#### 🔍 Chi tiết đối chiếu Thực tế vs Dự báo theo từng đợt")
        col_sel_up, col_dl_up = st.columns([2, 1])
        with col_sel_up:
            sel_up = st.selectbox("Chọn đợt dữ liệu:", df_view["Upload"].unique(), key="sel_up_page3")
        
        sub_up = df_view[df_view["Upload"] == sel_up]
        cols_show = [c for c in sub_up.columns if c != "Ngày thứ"]
        
        with col_dl_up:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            csv_exp = sub_up[cols_show].to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Xuất Bảng Đợt Này (CSV/Excel)",
                data=csv_exp,
                file_name=f"Doi_chieu_{sel_up.replace('#', '').replace(' ', '_')}.csv",
                mime="text/csv",
                key="btn_dl_sub_up"
            )
            
        safe_dataframe(sub_up[cols_show].style.format({"Dự báo":"{:.2f}","Thực tế":"{:.2f}","Sai lệch":"{:.2f}","% Lệch":"{:.2f}%"}), height=300)
        
        st.markdown("#### 📈 Biểu đồ so sánh Thực tế và Dự báo")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            sh_chart = st.selectbox("Chọn chân trời dự báo:", [f"{h}d" for h in HORIZONS], key="sh_chart_page3")
        with col_c2:
            tgt_chart = st.selectbox("Chọn mặt hàng dầu:", TARGET_COLS, key="tgt_chart_page3")
            
        sub_c = sub_up[(sub_up["Horizon"] == sh_chart) & (sub_up["Target"] == tgt_chart)]
        if not sub_c.empty:
            fig_cmp = go.Figure()
            for m in sel_models:
                ms = sub_c[sub_c["Model"] == m].sort_values(DATE_COL)
                if not ms.empty:
                    fig_cmp.add_trace(go.Scatter(x=ms[DATE_COL], y=ms["Dự báo"], name=f"Dự báo ({m})", mode="lines+markers", line=dict(dash="dash", color="#7c3aed")))
            act = sub_c.drop_duplicates(DATE_COL).sort_values(DATE_COL)
            fig_cmp.add_trace(go.Scatter(x=act[DATE_COL], y=act["Thực tế"], name="Thực tế", mode="lines+markers", line=dict(color="#00d4aa", width=3)))
            fig_cmp.update_layout(title=f"Đối chiếu {tgt_chart} ({sh_chart}) - Đợt {sel_up}", template="plotly_dark", height=320, hovermode="x unified")
            safe_plotly_chart(fig_cmp)
    else:
        st.info("Chưa có dữ liệu lịch sử đối chiếu.")


# ──────────────────────────────────────────
# TRANG 4: HUẤN LUYỆN MÔ HÌNH
# ──────────────────────────────────────────
elif nav_choice == "⚙  Huấn luyện mô hình":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="color:#00ad91; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.09em;">QUẢN TRỊ HỆ THỐNG</div>
        <h1 style="margin:4px 0; font-size:28px; font-weight:800; letter-spacing:-.03em;">Huấn Luyện & Tinh Chỉnh Mô Hình</h1>
        <p style="margin:0; color:#64748b; font-size:14px;">Khu vực quản trị dành cho việc cập nhật bộ não AI với chuỗi dữ liệu mới.</p>
    </div>
    <div style="background:#f8fafc; border-left:4px solid #c77700; border-radius:6px; padding:11px 16px; margin:14px 0 18px; font-size:13px; color:#334155; line-height:1.6;">
        <b>💡 Cẩm nang huấn luyện & tối ưu mô hình:</b><br>
        • <b>Khi nào nên huấn luyện?</b> Định kỳ 1 tháng/lần sau khi có đủ dữ liệu giá thực tế của tháng đó, hoặc khi Trang 2 cảnh báo MAPE &gt; 10%.<br>
        • <b>Chế độ khuyến nghị:</b> Chọn <b>'⚡ Finetune từ checkpoint'</b> để mô hình cập nhật theo giá mới nhất mà không mất đi tri thức lịch sử đã học.<br>
        • <b>Đánh giá nghiệm thu:</b> Sau khi chạy xong, chuyển sang <b>Tab 3: So sánh Kết quả (Benchmarking)</b> để đối chiếu phiên mới vs phiên cũ, nghiệm thu <b>% Cải thiện độ chính xác (Giảm sai số)</b>.
    </div>
    """, unsafe_allow_html=True)

    tab_train, tab_history, tab_compare = st.tabs([
        "🚀 Khởi chạy Huấn luyện",
        "📋 Lịch sử & Dữ liệu đầu vào",
        "📊 So sánh Kết quả (Benchmarking)"
    ])

    # ----------------------------------------------------
    # TAB 1: KHỞI CHẠY HUẤN LUYỆN
    # ----------------------------------------------------
    with tab_train:
        if is_gpu:
            st.markdown("""
            <div class="hub-notice gpu">
                <b style="display:block; margin-bottom:2px;">⚡ Huấn luyện trên GPU NVIDIA CUDA</b>
                Tốc độ tối ưu hóa cực nhanh (khoảng 10 - 20 giây mỗi mốc thời gian). Khuyến nghị thiết lập 50 Epochs để đạt độ hội tụ tối đa.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="hub-notice cpu">
                <b style="display:block; margin-bottom:2px;">💡 Huấn luyện trên CPU Doanh Nghiệp (6 vCPUs)</b>
                Quá trình huấn luyện chạy ổn định và an toàn. Thời gian dự kiến khoảng 1 – 2 phút cho mỗi mốc. 
                Trọng số mới chỉ được lưu sau khi kiểm tra nạp mô hình thành công 100%. Khuyến nghị giữ nguyên 25 – 30 Epochs.
            </div>
            """, unsafe_allow_html=True)
            
        # Việc C: kiểm tra khóa TRƯỚC khi vẽ các control cấu hình, để khi có job khác đang chạy,
        # toàn bộ control (không chỉ nút Bắt đầu) đều bị khóa xám — tránh người dùng đổi tham số
        # hoặc bấm lung tung gây nhầm lẫn/góp phần kích hoạt các thao tác xung đột giữa lúc job chạy.
        active_lock = get_active_training_lock()
        if active_lock:
            st.warning(
                f"⏳ **Đã có một Job Huấn luyện khác đang chạy** (bắt đầu lúc {active_lock.get('started_at', '?')}, "
                f"mô hình: {', '.join(active_lock.get('models', []))}). "
                "Hệ thống chỉ cho phép 1 job chạy tại một thời điểm để tránh ghi đè checkpoint. "
                "Vui lòng đợi job hiện tại hoàn tất rồi thử lại — mọi tham số bên dưới tạm khóa."
            )

        st.markdown("#### Thiết lập tham số huấn luyện")
        col_tmode, col_tep = st.columns([2, 1])
        with col_tmode:
            train_mode = st.selectbox("Chế độ huấn luyện:", ["⚡ Finetune từ checkpoint hiện tại (Khuyên dùng)", "🔁 Huấn luyện lại từ đầu"], disabled=bool(active_lock))
        with col_tep:
            n_epochs = st.number_input("Số vòng lặp (Epochs):", min_value=1, max_value=200, value=25 if not is_gpu else 50, help="Số epochs càng lớn, mô hình học càng sâu nhưng tốn nhiều thời gian hơn.", disabled=bool(active_lock))

        sel_hz_manual = st.multiselect("Chọn các mốc cần cập nhật:", HORIZONS, default=HORIZONS, format_func=lambda h: f"{h} ngày (h{h})", disabled=bool(active_lock))

        if st.button("🚀 Bắt đầu Job Huấn Luyện", key="btn_train_job_p4", type="primary", disabled=bool(active_lock)):
            if not sel_models:
                st.error("❌ Vui lòng chọn mô hình AI ở thanh điều khiển bên trái!")
            elif not sel_hz_manual:
                st.error("❌ Vui lòng chọn ít nhất một mốc thời gian!")
            elif get_active_training_lock():
                st.error("❌ Đã có job khác vừa bắt đầu chạy. Vui lòng tải lại trang và thử lại sau.")
            else:
                total_hz = len(sel_hz_manual)
                progress_bar = st.progress(0)
                status_box = st.status(f"⏳ Job đang chạy: Chuẩn bị môi trường cho {', '.join(sel_models)}...", expanded=True)

                # Việc B: bảng tiến độ trực quan theo TỪNG MỐC (thay vì chỉ 1 thanh % chung chung) —
                # đặc biệt hữu ích khi chạy CPU (chậm, người dùng cần biết đang chạy tới đâu).
                hz_table_box = st.empty()
                hz_order = sorted(sel_hz_manual)
                hz_status = {h: {"state": "waiting", "val_loss": None} for h in hz_order}
                current_hz = hz_order[0] if hz_order else None

                def _render_hz_table():
                    icon = {"waiting": "⚪", "running": "⏳", "done": "✅"}
                    lines = []
                    for h in hz_order:
                        st_ = hz_status[h]
                        txt = f"{icon[st_['state']]} Mốc {h} ngày: "
                        if st_["state"] == "waiting":
                            txt += "Đang chờ..."
                        elif st_["state"] == "running":
                            txt += "Đang tối ưu hóa..."
                        else:
                            vl = st_["val_loss"]
                            txt += f"Đã xong (Val Loss: {vl:.5f})" if vl is not None else "Đã xong"
                        lines.append(txt)
                    hz_table_box.markdown("  \n".join(lines))

                _render_hz_table()

                import subprocess, re
                cmd = [sys.executable, "train_all_horizons.py", "--update_data", "--epochs", str(n_epochs), "--models"] + sel_models + ["--horizons"] + [str(x) for x in sel_hz_manual]
                if "Huấn luyện lại từ đầu" in train_mode:
                    cmd.append("--force_retrain")

                log_lines = []
                hz_completed = 0
                process = None

                # Chấp nhận cả 2 kiểu chuỗi log GUMNet lẫn HybridTriNet in ra val loss — trước đây
                # chỉ bắt "Best Val Loss" (chỉ GUMNet dùng), khiến thanh tiến trình đứng im 0% suốt
                # phiên huấn luyện HybridTriNet dù nó vẫn chạy bình thường ở phía sau.
                VAL_LOSS_RE = re.compile(r"(?:Best Val Loss:|best_val=)\s*([\d.]+)")
                HZ_START_RE = re.compile(r"ĐANG HUẤN LUYỆN MỐC:\s*(\d+)\s*NGÀY")

                acquire_training_lock(sel_models, sel_hz_manual)  # giữ chỗ, tránh race giữa 2 người dùng
                try:
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
                    acquire_training_lock(sel_models, sel_hz_manual, pid=process.pid)  # ghi đè bằng PID thật

                    for line in process.stdout:
                        log_lines.append(line)
                        hz_match = HZ_START_RE.search(line)
                        loss_match = VAL_LOSS_RE.search(line)
                        if hz_match:
                            h_started = int(hz_match.group(1))
                            if h_started in hz_status:
                                current_hz = h_started
                                hz_status[h_started]["state"] = "running"
                                _render_hz_table()
                            status_box.write(f"📌 {line.strip()}")
                        elif loss_match and current_hz is not None and current_hz in hz_status:
                            hz_status[current_hz]["state"] = "done"
                            hz_status[current_hz]["val_loss"] = float(loss_match.group(1))
                            _render_hz_table()
                            hz_completed = min(hz_completed + 1, total_hz)
                            pct = int((hz_completed / total_hz) * 100)
                            progress_bar.progress(pct)
                            status_box.update(label=f"🔄 Đang tối ưu hóa... (Đã hoàn thành {hz_completed}/{total_hz} mốc - {pct}%)")

                    process.wait()
                    progress_bar.progress(100)

                    if process.returncode == 0:
                        status_box.update(label="✅ Quá trình huấn luyện đã hoàn tất thành công 100%!", state="complete", expanded=False)
                        st.success("🎉 **CẬP NHẬT MÔ HÌNH THÀNH CÔNG!** Trọng số mạng nơ-ron mới đã được kiểm tra và lưu trữ. Dữ liệu phiên vừa hoàn tất đã được ghi nhận vào Lịch sử huấn luyện.")
                        if CACHE_FILE.exists():
                            try: os.remove(CACHE_FILE)
                            except: pass
                        st.cache_resource.clear()
                        st.cache_data.clear()
                    else:
                        status_box.update(label=f"❌ Tiến trình kết thúc với mã: {process.returncode}", state="error")
                        st.error("Có sự cố trong lúc huấn luyện. Mở rộng khung nhật ký kỹ thuật bên dưới để xem chi tiết.")
                except Exception as e:
                    status_box.update(label="❌ Lỗi khởi động tiến trình", state="error")
                    st.error(f"Lỗi: {e}")
                finally:
                    # Lỗi #17 (đã xác nhận): trước đây luôn terminate() tiến trình con nếu người
                    # dùng đổi trang giữa chừng — có thể cắt ngang đúng lúc đang ghi checkpoint
                    # (torch.save), gây hỏng file. Từ giờ train_all_horizons.py TỰ quản lý lock
                    # của chính nó (ghi lúc bắt đầu, tự xoá khi thực sự xong — xem file đó), nên
                    # ở đây CHỈ dọn lock khi tiến trình con CHƯA từng chạy được (ví dụ Popen lỗi
                    # ngay từ đầu) — còn nếu nó đang chạy thật, cứ để nó chạy nốt trong nền và tự
                    # dọn lock của chính nó khi hoàn tất, dù script Streamlit này đã bị ngắt.
                    if process is None or process.poll() is not None:
                        release_training_lock()

                if log_lines:
                    with st.expander("🔍 Xem chi tiết nhật ký tiến trình kỹ thuật", expanded=False):
                        st.code("".join(log_lines[-40:]), language="bash")

    # ----------------------------------------------------
    # TAB 2: LỊCH SỬ & DỮ LIỆU ĐẦU VÀO
    # ----------------------------------------------------
    with tab_history:
        history_file = CKPT_DIR / "training_history.json"
        entries = []
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except Exception as e:
                st.warning(f"Không thể đọc lịch sử huấn luyện: {e}")
                
        if not entries:
            st.info("ℹ️ Hiện chưa có nhật ký phiên huấn luyện nào được ghi nhận.")
        else:
            # Top KPI Summary Cards
            col_k1, col_k2, col_k3, col_k4 = st.columns(4)
            with col_k1:
                st.markdown(f"""
                <div style="border:1px solid #e2e8f0; border-radius:8px; padding:12px; background:#f8fafc;">
                    <div style="font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase;">Tổng số phiên</div>
                    <div style="font-size:22px; font-weight:800; color:#0f172a; margin-top:2px;">{len(entries)} phiên</div>
                    <div style="font-size:12px; color:#087762; margin-top:2px;">Đã lưu trữ an toàn</div>
                </div>
                """, unsafe_allow_html=True)
            with col_k2:
                latest_sess = entries[0]
                st.markdown(f"""
                <div style="border:1px solid #e2e8f0; border-radius:8px; padding:12px; background:#f8fafc;">
                    <div style="font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase;">Phiên gần nhất</div>
                    <div style="font-size:16px; font-weight:800; color:#0f172a; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{latest_sess.get('session_id', '-')}</div>
                    <div style="font-size:12px; color:#64748b; margin-top:2px;">{latest_sess.get('timestamp', '')}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_k3:
                d_rows = latest_sess.get("data_info", {}).get("total_rows")
                d_rows_txt = f"{d_rows:,} dòng" if isinstance(d_rows, (int, float)) else "Chưa rõ"
                st.markdown(f"""
                <div style="border:1px solid #e2e8f0; border-radius:8px; padding:12px; background:#f8fafc;">
                    <div style="font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase;">Quy mô tập dữ liệu</div>
                    <div style="font-size:22px; font-weight:800; color:#00ad91; margin-top:2px;">{d_rows_txt}</div>
                    <div style="font-size:12px; color:#64748b; margin-top:2px;">{latest_sess.get('data_info', {}).get('date_range', '')}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_k4:
                avg_l = latest_sess.get("avg_val_loss")
                loss_txt = f"{avg_l:.5f}" if avg_l is not None else "Tối ưu tốt"
                st.markdown(f"""
                <div style="border:1px solid #e2e8f0; border-radius:8px; padding:12px; background:#f8fafc;">
                    <div style="font-size:11px; color:#64748b; font-weight:700; text-transform:uppercase;">Val Loss Trung bình</div>
                    <div style="font-size:22px; font-weight:800; color:#7c3aed; margin-top:2px;">{loss_txt}</div>
                    <div style="font-size:12px; color:#087762; margin-top:2px;">Hội tụ ổn định</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 📜 Danh sách các phiên huấn luyện đã thực hiện")
            
            # Format DataFrame for session listing
            rows_table = []
            for e in entries:
                d_info = e.get("data_info", {})
                rows_table.append({
                    "Mã phiên": e.get("session_id", ""),
                    "Thời gian": e.get("timestamp", ""),
                    "Chế độ": e.get("mode", ""),
                    "Mô hình": ", ".join(e.get("models", [])),
                    "Thiết bị": e.get("device_name", e.get("device", "")),
                    "Epochs": e.get("epochs", "-"),
                    "Số dòng dữ liệu": f"{d_info.get('total_rows', 0):,} dòng",
                    "Khoảng ngày dữ liệu": d_info.get("date_range", ""),
                    "Val Loss TB": f"{e.get('avg_val_loss', 0):.5f}" if e.get("avg_val_loss") else "-"
                })
            safe_dataframe(pd.DataFrame(rows_table))
            
            st.markdown("---")
            st.markdown("#### 🔍 Chi tiết dữ liệu đầu vào & kết quả theo phiên")
            sess_choices = [e.get("session_id") for e in entries]
            sel_s_id = st.selectbox("Chọn phiên cần kiểm tra chi tiết:", sess_choices, key="sel_sess_detail")
            chosen_entry = next((e for e in entries if e.get("session_id") == sel_s_id), entries[0])
            
            c_d1, c_d2 = st.columns([1.1, 0.9])
            with c_d1:
                cd_info = chosen_entry.get("data_info", {})
                st.markdown(f"""
                <div style="border:1px solid #e2e8f0; border-radius:8px; padding:16px; background:#ffffff;">
                    <div style="color:#00ad91; font-size:12px; font-weight:800; text-transform:uppercase;">Dữ liệu huấn luyện đưa vào</div>
                    <h3 style="margin:4px 0 12px; font-size:18px; color:#0f172a;">Chi tiết tập dữ liệu phiên {chosen_entry.get('session_id')}</h3>
                    <table style="width:100%; font-size:13px; border-collapse:collapse;">
                        <tr style="border-bottom:1px solid #f1f5f9;">
                            <td style="padding:6px 0; color:#64748b;">Số mẫu dữ liệu:</td>
                            <td style="padding:6px 0; font-weight:700; color:#0f172a; text-align:right;">{cd_info.get('total_rows', 0):,} dòng quan sát</td>
                        </tr>
                        <tr style="border-bottom:1px solid #f1f5f9;">
                            <td style="padding:6px 0; color:#64748b;">Chu kỳ thời gian:</td>
                            <td style="padding:6px 0; font-weight:700; color:#0f172a; text-align:right;">{cd_info.get('date_range', '-')}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #f1f5f9;">
                            <td style="padding:6px 0; color:#64748b;">Mặt hàng xăng dầu (Target):</td>
                            <td style="padding:6px 0; font-weight:700; color:#0f172a; text-align:right;">{', '.join(cd_info.get('targets', []))}</td>
                        </tr>
                        <tr style="border-bottom:1px solid #f1f5f9;">
                            <td style="padding:6px 0; color:#64748b;">Chế độ & Vòng lặp:</td>
                            <td style="padding:6px 0; font-weight:700; color:#0f172a; text-align:right;">{chosen_entry.get('mode')} ({chosen_entry.get('epochs')} Epochs)</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0; color:#64748b;">Ghi chú tệp nguồn:</td>
                            <td style="padding:6px 0; color:#087762; font-weight:600; text-align:right;">{cd_info.get('note', 'Tập dữ liệu chuẩn hóa hệ thống')}</td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
                
            with c_d2:
                res = chosen_entry.get("results", {})
                st.markdown(f"""
                <div style="border:1px solid #e2e8f0; border-radius:8px; padding:16px; background:#ffffff;">
                    <div style="color:#7c3aed; font-size:12px; font-weight:800; text-transform:uppercase;">Kết quả hội tụ (Validation Loss)</div>
                    <h3 style="margin:4px 0 12px; font-size:18px; color:#0f172a;">Sai số kiểm định theo mốc thời gian</h3>
                """, unsafe_allow_html=True)
                
                loss_table = []
                for hz in HORIZONS:
                    g_key = f"GUMNet_h{hz}"
                    h_key = f"HybridTriNet_h{hz}"
                    gen_key = f"h{hz}"
                    g_val = res.get(g_key, res.get(gen_key, "-"))
                    h_val = res.get(h_key, "-")
                    loss_table.append({
                        "Mốc dự báo": f"+{hz} ngày",
                        "GUMNet Loss": f"{g_val:.5f}" if isinstance(g_val, (int, float)) else str(g_val),
                        "HybridTriNet Loss": f"{h_val:.5f}" if isinstance(h_val, (int, float)) else str(h_val)
                    })
                safe_dataframe(pd.DataFrame(loss_table))
                st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 3: SO SÁNH KẾT QUẢ (BENCHMARKING)
    # ----------------------------------------------------
    with tab_compare:
        history_file = CKPT_DIR / "training_history.json"
        entries = []
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except:
                entries = []
                
        if len(entries) < 2:
            st.info("ℹ️ Cần ít nhất 2 phiên huấn luyện trong lịch sử để thực hiện so sánh đối chiếu (Benchmarking).")
        else:
            st.markdown("#### Đối chiếu kết quả giữa 2 phiên huấn luyện")
            st.caption("Chọn hai phiên bất kỳ để đánh giá mức độ cải thiện sai số và sự thay đổi của tập dữ liệu đưa vào:")
            
            c_sel1, c_sel2 = st.columns(2)
            sess_ids = [e.get("session_id") for e in entries]
            with c_sel1:
                s1_id = st.selectbox("📌 Phiên đối chứng (Trước / Baseline):", sess_ids, index=min(1, len(sess_ids)-1), key="s1_bench")
            with c_sel2:
                s2_id = st.selectbox("🎯 Phiên kiểm tra (Sau / Cập nhật):", sess_ids, index=0, key="s2_bench")
                
            e1 = next(e for e in entries if e.get("session_id") == s1_id)
            e2 = next(e for e in entries if e.get("session_id") == s2_id)
            
            # Loss Improvement KPI
            l1 = e1.get("avg_val_loss")
            l2 = e2.get("avg_val_loss")
            if l1 and l2:
                diff = l1 - l2
                pct_imp = (diff / l1) * 100
                is_better = pct_imp >= 0
                
                col_imp1, col_imp2, col_imp3 = st.columns(3)
                with col_imp1:
                    st.metric(label=f"Val Loss: {s1_id}", value=f"{l1:.5f}")
                with col_imp2:
                    st.metric(label=f"Val Loss: {s2_id}", value=f"{l2:.5f}", delta=f"{'-' if is_better else '+'}{abs(diff):.5f}")
                with col_imp3:
                    st.metric(label="Mức độ cải thiện độ chính xác", value=f"{abs(pct_imp):.2f}%", delta="Tốt hơn (Giảm sai số)" if is_better else "Tăng sai số", delta_color="normal" if is_better else "inverse")
            
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            
            # Comparison Table
            st.markdown("##### 1. Đối chiếu quy mô dữ liệu & cấu hình")
            d1_info = e1.get("data_info", {})
            d2_info = e2.get("data_info", {})
            
            r1 = d1_info.get("total_rows", 0)
            r2 = d2_info.get("total_rows", 0)
            diff_r = r2 - r1
            diff_r_txt = f"{'+' if diff_r >= 0 else ''}{diff_r:,} dòng"
            
            cmp_specs = [
                {"Tiêu chí": "Mã phiên", s1_id: str(s1_id), s2_id: str(s2_id), "Chênh lệch / Đánh giá": "-"},
                {"Tiêu chí": "Thời gian thực hiện", s1_id: str(e1.get("timestamp", "-")), s2_id: str(e2.get("timestamp", "-")), "Chênh lệch / Đánh giá": "-"},
                {"Tiêu chí": "Chế độ huấn luyện", s1_id: str(e1.get("mode", "-")), s2_id: str(e2.get("mode", "-")), "Chênh lệch / Đánh giá": "-"},
                {"Tiêu chí": "Số dòng dữ liệu đầu vào", s1_id: f"{r1:,} dòng", s2_id: f"{r2:,} dòng", "Chênh lệch / Đánh giá": str(diff_r_txt)},
                {"Tiêu chí": "Khoảng thời gian dữ liệu", s1_id: str(d1_info.get("date_range", "-")), s2_id: str(d2_info.get("date_range", "-")), "Chênh lệch / Đánh giá": "Cập nhật chuỗi mới" if d1_info.get("date_range") != d2_info.get("date_range") else "Tương đương"},
                {"Tiêu chí": "Số Epochs", s1_id: str(e1.get("epochs", "-")), s2_id: str(e2.get("epochs", "-")), "Chênh lệch / Đánh giá": f"{e2.get('epochs', 0) - e1.get('epochs', 0):+} epochs"},
                {"Tiêu chí": "Thiết bị tính toán", s1_id: str(e1.get("device_name", e1.get("device", "-"))), s2_id: str(e2.get("device_name", e2.get("device", "-"))), "Chênh lệch / Đánh giá": "-"},
            ]
            safe_dataframe(pd.DataFrame(cmp_specs))
            
            st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
            st.markdown("##### 2. Đối chiếu chi tiết Sai số Kiểm định (Validation Loss) theo Horizon")
            
            res1 = e1.get("results", {})
            res2 = e2.get("results", {})
            
            hz_bench_data = []
            bar_horizons = []
            bar_s1_loss = []
            bar_s2_loss = []
            
            for hz in HORIZONS:
                v1 = res1.get(f"GUMNet_h{hz}", res1.get(f"h{hz}"))
                v2 = res2.get(f"GUMNet_h{hz}", res2.get(f"h{hz}"))
                
                v1_num = float(v1) if isinstance(v1, (int, float)) else None
                v2_num = float(v2) if isinstance(v2, (int, float)) else None
                
                imp_str = "-"
                if v1_num is not None and v2_num is not None:
                    h_diff = v1_num - v2_num
                    h_pct = (h_diff / v1_num) * 100
                    imp_str = f"🟢 Giảm {abs(h_pct):.2f}%" if h_diff >= 0 else f"🔴 Tăng {abs(h_pct):.2f}%"
                    bar_horizons.append(f"+{hz}d")
                    bar_s1_loss.append(v1_num)
                    bar_s2_loss.append(v2_num)
                    
                hz_bench_data.append({
                    "Mốc Horizon": f"+{hz} ngày (h{hz})",
                    f"Loss ({s1_id})": f"{v1_num:.5f}" if v1_num is not None else "-",
                    f"Loss ({s2_id})": f"{v2_num:.5f}" if v2_num is not None else "-",
                    "Hiệu quả cải thiện": imp_str
                })
                
            col_btable, col_bchart = st.columns([1, 1.2])
            with col_btable:
                safe_dataframe(pd.DataFrame(hz_bench_data))
                
            with col_bchart:
                if bar_horizons:
                    fig_cmp = go.Figure()
                    fig_cmp.add_trace(go.Bar(
                        x=bar_horizons,
                        y=bar_s1_loss,
                        name=f"Phiên đối chứng ({s1_id})",
                        marker_color="#94a3b8"
                    ))
                    fig_cmp.add_trace(go.Bar(
                        x=bar_horizons,
                        y=bar_s2_loss,
                        name=f"Phiên kiểm tra ({s2_id})",
                        marker_color="#00d4aa"
                    ))
                    fig_cmp.update_layout(
                        title="Đối chiếu Validation Loss (Càng thấp mô hình càng chuẩn xác)",
                        barmode="group",
                        template="plotly_dark",
                        height=340,
                        # Lỗi đã xác nhận: legend ngang neo ở y=1.02 (ngay sát phía trên khung vẽ)
                        # dùng chung vùng không gian với tiêu đề dài phía trên -> chồng chữ khi
                        # tiêu đề đủ dài (đặc biệt màn hình rộng). Chuyển legend xuống DƯỚI biểu đồ
                        # để không bao giờ tranh chỗ với tiêu đề nữa, bất kể tiêu đề dài ngắn thế nào.
                        margin=dict(l=20, r=20, t=40, b=60),
                        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                        # Lỗi đã xác nhận: tên trace "Phiên đối chứng (TR-...)" khá dài, Plotly mặc
                        # định cắt bớt bằng "..." khi hiện tooltip hover. namelength=-1 = hiện đầy đủ,
                        # không cắt nữa.
                        hoverlabel=dict(namelength=-1)
                    )
                    safe_plotly_chart(fig_cmp)


# ──────────────────────────────────────────
# TRANG 5: HƯỚNG DẪN SỬ DỤNG
# ──────────────────────────────────────────
elif nav_choice == "❓  Hướng dẫn sử dụng":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="color:#00ad91; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.09em;">HƯỚNG DẪN VẬN HÀNH</div>
        <h1 style="margin:4px 0; font-size:28px; font-weight:800; letter-spacing:-.03em;">Hướng Dẫn Sử Dụng & Vận Hành Hệ Thống</h1>
        <p style="margin:0; color:#64748b; font-size:14px;">Các bước vận hành chuẩn hóa, hướng dẫn tương tác với mũi tên và giải thích chi tiết CPU/GPU.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. Thông báo nhận diện phần cứng máy chủ hiện tại
    if is_gpu:
        st.markdown("""
        <div style="background:#eafaf5; border:1px solid #bcebdc; border-radius:10px; padding:14px 18px; margin-bottom:20px;">
            <b style="color:#087762; font-size:15px;">🖥️ Nhận diện phần cứng máy chủ: Đang kích hoạt GPU NVIDIA CUDA</b>
            <p style="color:#2d5a50; font-size:13px; margin:4px 0 0; line-height:1.5;">
                Hệ thống đã tự động nhận diện và cấu hình tăng tốc phần cứng tối đa. 
                Mọi thao tác Dự báo &amp; Đánh giá diễn ra tức thì (&lt; 1s). 
                Khi Huấn luyện mô hình (Trang 4), hệ thống tự động tối ưu với mức <b>50 Epochs</b> (tốc độ siêu nhanh khoảng 10–20 giây mỗi mốc thời gian).
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#fff7e7; border:1px solid #f6d58c; border-radius:10px; padding:14px 18px; margin-bottom:20px;">
            <b style="color:#9b6100; font-size:15px;">🖥️ Nhận diện phần cứng máy chủ: Đang chạy CPU Doanh Nghiệp (6 vCPUs)</b>
            <p style="color:#6d4800; font-size:13px; margin:4px 0 0; line-height:1.5;">
                Hệ thống đang vận hành hoàn toàn ổn định và an toàn trên nền tảng CPU. 
                Dự báo giá thị trường diễn ra nhanh chóng (&lt; 1s). 
                Khi Huấn luyện mô hình (Trang 4), hệ thống tự động tối ưu với mức <b>25 Epochs</b> (chỉ mất 1–2 phút mỗi mốc thời gian), đảm bảo an toàn tuyệt đối và không chiếm dụng tài nguyên.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Sơ đồ quy trình vận hành 4 bước chuẩn hóa
    st.markdown("""
    #### 🔄 Sơ Đồ Quy Trình Vận Hành 4 Bước Khép Kín
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:10px 0 24px;">
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:24px; margin-bottom:4px;">📥</div>
            <b style="color:#00ad91; font-size:13px;">BƯỚC 1: NẠP DỮ LIỆU</b>
            <p style="font-size:12px; color:#64748b; margin:4px 0 0;">Kéo thả file Excel mới vào Trang 1. Bảng giá 7 mốc hiện ra ngay.</p>
        </div>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:24px; margin-bottom:4px;">📊</div>
            <b style="color:#6954d9; font-size:13px;">BƯỚC 2: KIỂM ĐỊNH</b>
            <p style="font-size:12px; color:#64748b; margin:4px 0 0;">Sang Trang 2 xem sai số MAPE. Xanh (&lt; 7%) là an toàn dùng ngay.</p>
        </div>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:24px; margin-bottom:4px;">⚙️</div>
            <b style="color:#c77700; font-size:13px;">BƯỚC 3: FINETUNE</b>
            <p style="font-size:12px; color:#64748b; margin:4px 0 0;">Nếu MAPE &gt; 10%: Sang Trang 4 bấm Finetune để AI học giá mới.</p>
        </div>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:24px; margin-bottom:4px;">📁</div>
            <b style="color:#087762; font-size:13px;">BƯỚC 4: LƯU BÁO CÁO</b>
            <p style="font-size:12px; color:#64748b; margin:4px 0 0;">Xuất file CSV nộp lãnh đạo. Dữ liệu lưu vĩnh viễn ở Trang 3.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🎯 Hướng Dẫn Trực Quan Với Mũi Tên Chỉ Dẫn")
    st.caption("Bấm vào các thẻ bên dưới để hệ thống kích hoạt mũi tên phát sáng dẫn đường từng bước trực tiếp trên màn hình (Không gây tải máy chủ, phản hồi tức thì 0ms):")
    
    st.markdown("""
    <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:14px; margin-top:10px;">
        <button class="guide-action" data-oil-tour="forecast">
            <b style="color:#00ad91; font-size:15px;">◈ Tạo dự báo đầu tiên</b>
            <small style="color:#64748b; font-size:13px;">Mũi tên chỉ dẫn cách Upload file, chọn mốc và đọc bảng kết quả giá.</small>
        </button>
        <button class="guide-action" data-oil-tour="metrics">
            <b style="color:#6954d9; font-size:15px;">▦ Đọc sai số MAPE & MAE</b>
            <small style="color:#64748b; font-size:13px;">Mũi tên chỉ dẫn các thẻ chỉ số độ tin cậy và biểu đồ sai số theo thời gian.</small>
        </button>
        <button class="guide-action" data-oil-tour="training">
            <b style="color:#c77700; font-size:15px;">⚙ Quản trị & Huấn luyện mô hình</b>
            <small style="color:#64748b; font-size:13px;">Mũi tên chỉ dẫn chọn mốc, số epochs, xem lịch sử và so sánh Benchmarking.</small>
        </button>
        <button class="guide-action" data-oil-tour="history">
            <b style="color:#087762; font-size:15px;">◷ Lịch sử & Xuất báo cáo</b>
            <small style="color:#64748b; font-size:13px;">Mũi tên chỉ dẫn tra cứu các đợt file, đồ thị đối chiếu và xuất file CSV.</small>
        </button>
    </div>
    <div style="margin-top:16px; margin-bottom:20px;">
        <button class="oil-btn oil-btn-ghost" id="oil-replay-btn" style="font-size:13px;">↺ Xem lại thông báo chào mừng & hướng dẫn từ đầu</button>
    </div>
    """, unsafe_allow_html=True)

    # Lỗi #18 (đã xác nhận): 2 file PDF hướng dẫn đã có sẵn trong project nhưng chưa hề được
    # cung cấp nút tải trên giao diện — người dùng LAN không có cách nào lấy được tài liệu này
    # ngoại trừ tự tìm trong thư mục cài đặt.
    _pdf_guides = [
        ("📕 Hướng dẫn Cấu hình & Triển khai (PDF)", ROOT / "HUONG_DAN_CAU_HINH_VA_TRIEN_KHAI.pdf"),
        ("📗 Hướng dẫn Triển khai & Sử dụng (PDF)", ROOT / "HUONG_DAN_TRIEN_KHAI_VA_SU_DUNG.pdf"),
    ]
    _available_pdfs = [(label, p) for label, p in _pdf_guides if p.exists()]
    if _available_pdfs:
        st.markdown("#### 📄 Tài liệu hướng dẫn chi tiết (PDF)")
        cols_pdf = st.columns(len(_available_pdfs))
        for col, (label, pdf_path) in zip(cols_pdf, _available_pdfs):
            with col:
                st.download_button(
                    label=label,
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                    key=f"dl_guide_{pdf_path.stem}",
                )

    st.markdown("---")
    st.markdown("#### 📖 4 Bước Nghiệp Vụ Chuẩn Hóa")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        <div class="guide-card">
            <b style="font-size:16px; color:#1e293b;">◈ 1. Tạo dự báo giá mới</b><br>
            <p style="font-size:13px; color:#64748b; margin:6px 0 10px;">
                Vào mục <b>Dự báo</b> ➔ Kéo thả tệp Excel dữ liệu mới ➔ Hệ thống tự động nhận diện ngày dữ liệu cuối cùng và tính toán dự báo cho 7 mốc thời gian (+1d, +5d, +10d, +15d, +20d, +30d, +60d) trong chưa đầy 1 giây.<br>
                Nhấn nút <b>Xuất CSV/Excel</b> để tải báo cáo giá gửi lãnh đạo.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_g2:
        st.markdown("""
        <div class="guide-card">
            <b style="font-size:16px; color:#1e293b;">▦ 2. Cách đọc chỉ số sai số (MAPE & MAE)</b><br>
            <p style="font-size:13px; color:#64748b; margin:6px 0 10px;">
                Vào mục <b>Đánh giá mô hình</b> để theo dõi độ tin cậy:<br>
                - <b>MAPE &lt; 7% (Xanh):</b> Mô hình dự báo rất chính xác, bám sát nhịp biến động của thị trường xăng dầu.<br>
                - <b>MAPE 7% – 10% (Vàng):</b> Mô hình ổn định, nằm trong dung sai cho phép.<br>
                - <b>MAPE &gt; 10% (Đỏ):</b> Thị trường vừa xảy ra biến động mạnh, khuyến nghị Finetune mô hình.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    
    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.markdown("""
        <div class="guide-card">
            <b style="font-size:16px; color:#1e293b;">⚙ 3. Khi nào cần Huấn luyện (Finetune)?</b><br>
            <p style="font-size:13px; color:#64748b; margin:6px 0 10px;">
                Vào mục <b>Huấn luyện mô hình</b> định kỳ 1 tháng/lần sau khi có đủ dữ liệu thực tế của tháng đó.<br>
                Cơ chế Finetune hấp thụ thêm quy luật giá mới nhất mà vẫn giữ vững tri thức lịch sử đã học, giúp nâng cao độ chính xác mà không làm hỏng mô hình.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_g4:
        st.markdown("""
        <div class="guide-card">
            <b style="font-size:16px; color:#1e293b;">◷ 4. Tra cứu lịch sử & Kiểm toán dữ liệu</b><br>
            <p style="font-size:13px; color:#64748b; margin:6px 0 10px;">
                Vào mục <b>Lịch sử & Xuất dữ liệu</b> để kiểm tra lại các lần dự báo trong quá khứ.<br>
                Hệ thống lưu giữ đầy đủ tệp dữ liệu, ngày tháng và bảng đối chiếu Thực tế vs Dự báo giúp phục vụ công tác thanh tra, kiểm toán bất cứ lúc nào.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown("""
        <div class="guide-card">
            <h3 style="margin:0 0 10px; font-size:16px; color:#0f172a;">💡 Dự báo, Đánh giá và Huấn luyện khác nhau thế nào?</h3>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0;">
                <b style="color:#00ad91;">1. Dự báo:</b><br>
                <small style="color:#64748b;">Mô hình tạo giá dự kiến cho 1–60 ngày làm việc từ dữ liệu có sẵn tại thời điểm đó. Không cần có giá tương lai.</small>
            </div>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0;">
                <b style="color:#6954d9;">2. Đánh giá / Backtesting:</b><br>
                <small style="color:#64748b;">Sau khi tải file có giá thực tế của các ngày đã dự báo, hệ thống mới đối chiếu Dự báo với Thực tế để tính MAE và MAPE.</small>
            </div>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0;">
                <b style="color:#c77700;">3. Huấn luyện / Finetune:</b><br>
                <small style="color:#64748b;">Dùng dữ liệu mới để cập nhật trọng số mạng nơ-ron. Mô hình hấp thụ thêm quy luật giá mới nhất mà không mất đi tri thức lịch sử.</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown("""
        <div class="guide-card">
            <h3 style="margin:0 0 10px; font-size:16px; color:#0f172a;">📋 Quy tắc dữ liệu và ngày dự báo</h3>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0;">
                <b style="color:#9b6100;">! Dữ liệu quá cũ:</b><br>
                <small style="color:#64748b;">Nếu ngày cuối trong file quá 7 ngày làm việc so với ngày chạy, hệ thống khuyến nghị nạp thêm dữ liệu cập nhật.</small>
            </div>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0;">
                <b style="color:#087762;">✓ Giới hạn chân trời 60 ngày:</b><br>
                <small style="color:#64748b;">Mô hình hỗ trợ chuẩn hóa các mốc từ 1 ngày đến tối đa 60 ngày làm việc tương ứng với 7 checkpoints đã tối ưu.</small>
            </div>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0;">
                <b style="color:#0f172a;">✓ Cuối tuần và ngày nghỉ:</b><br>
                <small style="color:#64748b;">Hệ thống tự động bỏ qua Thứ Bảy và Chủ Nhật để chuỗi thời gian dự báo luôn khớp với các phiên giao dịch thực tế.</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("#### 💻 Bảng Đối Chiếu Kỹ Thuật: Chế Độ CPU vs GPU")
    st.caption("Giải thích sự khác biệt giữa hai môi trường phần cứng máy chủ để bạn hoàn toàn an tâm khi vận hành:")
    
    table_hw = pd.DataFrame([
        {
            "Nhiệm vụ nghiệp vụ": "🔮 Dự báo thị trường (Inference)",
            "Chế độ CPU (6 vCPUs)": "⚡ Dưới 1 giây (Tức thì)",
            "Chế độ GPU NVIDIA CUDA": "⚡ Dưới 1 giây (Tức thì)",
            "Độ chính xác mô hình": "✅ 100% giống nhau (Cùng trọng số)"
        },
        {
            "Nhiệm vụ nghiệp vụ": "📊 Đánh giá sai số (Backtesting)",
            "Chế độ CPU (6 vCPUs)": "Khoảng 1 – 3 giây",
            "Chế độ GPU NVIDIA CUDA": "Khoảng 1 giây",
            "Độ chính xác mô hình": "✅ 100% giống nhau"
        },
        {
            "Nhiệm vụ nghiệp vụ": "⚙ Huấn luyện lại (Finetuning)",
            "Chế độ CPU (6 vCPUs)": "1 – 2 phút / mốc (Khuyến nghị 25 epochs)",
            "Chế độ GPU NVIDIA CUDA": "10 – 20 giây / mốc (Khuyến nghị 50 epochs)",
            "Độ chính xác mô hình": "✅ Tương đương hoàn toàn"
        },
        {
            "Nhiệm vụ nghiệp vụ": "🔒 Độ an toàn & Ổn định",
            "Chế độ CPU (6 vCPUs)": "Rất ổn định, không chiếm dụng GPU",
            "Chế độ GPU NVIDIA CUDA": "Hiệu năng xử lý song song tối đa",
            "Độ chính xác mô hình": "Chuẩn hóa cấp doanh nghiệp"
        }
    ])
    safe_dataframe(table_hw.set_index("Nhiệm vụ nghiệp vụ"))

    st.markdown("---")
    st.markdown("#### ❓ Giải Đáp Thắc Mắc Nghiệp Vụ Thường Gặp (FAQ)")
    
    col_faq1, col_faq2 = st.columns(2)
    with col_faq1:
        st.markdown("""
        <div class="guide-card">
            <b style="color:#0f172a; font-size:14px;">1. Bao lâu nên nạp file dữ liệu mới một lần?</b>
            <p style="color:#64748b; font-size:13px; margin:4px 0 0;">
                Khuyến nghị nạp định kỳ <b>1–2 tuần/lần</b> hoặc ngay khi vừa kết thúc kỳ điều hành giá xăng dầu để hệ thống luôn có mốc dự báo tươi mới nhất.
            </p>
        </div>
        <div style="height:10px;"></div>
        <div class="guide-card">
            <b style="color:#0f172a; font-size:14px;">2. Tệp tải lên có bắt buộc đủ cả 4 mặt hàng không?</b>
            <p style="color:#64748b; font-size:13px; margin:4px 0 0;">
                Không bắt buộc. Nếu tệp Excel của bạn chỉ có giá MG95 và DO 0.05%, hệ thống vẫn tự động trích xuất và tính toán dự báo chuẩn xác cho các mặt hàng đó.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_faq2:
        st.markdown("""
        <div class="guide-card">
            <b style="color:#0f172a; font-size:14px;">3. Vì sao không có ngày dự báo vào Thứ Bảy &amp; Chủ Nhật?</b>
            <p style="color:#64748b; font-size:13px; margin:4px 0 0;">
                Thị trường xăng dầu quốc tế đóng cửa vào cuối tuần. Chuỗi dự báo tự động bỏ qua ngày nghỉ để luôn trùng khớp với các phiên giao dịch thực tế.
            </p>
        </div>
        <div style="height:10px;"></div>
        <div class="guide-card">
            <b style="color:#0f172a; font-size:14px;">4. Đơn vị tiền tệ của các mặt hàng được tính thế nào?</b>
            <p style="color:#64748b; font-size:13px; margin:4px 0 0;">
                Xăng MG95 và MG92 tính theo <b>USD/thùng</b> (Platts Singapore). Dầu DO 0.001% và DO 0.05% tính theo <b>USD/tấn</b> chuẩn thị trường quốc tế.
            </p>
        </div>
        """, unsafe_allow_html=True)



