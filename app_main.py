"""
Multi-Model Oil Price Forecast – Evaluation Hub (Robust Version)
"""

import sys, json, importlib, warnings, os, logging, uuid, subprocess, re, time, hashlib, html

# Tắt toàn bộ cảnh báo (scikit-learn version, streamlit deprecation, etc.) để Terminal luôn sạch đẹp
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
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

from project_io import load_checkpoint, read_cache, write_cache, save_upload, dataset_fingerprint, process_alive
import data_pipeline
import pipeline_engine

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
                    text: "Kéo thả file Excel (.xlsx) hoặc CSV vào đây. Xem kết quả kiểm tra file, sau đó bấm nút cập nhật để hệ thống xử lý dữ liệu và tạo dự báo 7 mốc.",
                    selector: '[data-testid="stFileUploader"], .stFileUploader'
                },
                {
                    title: "2. Giám sát trạng thái & phần cứng",
                    text: "Kiểm tra phiên bản dữ liệu gần nhất, thiết bị tính toán (GPU/CPU) và tình trạng nạp sẵn sàng 7 mốc dự báo (+1d đến +60d) từ mô hình GUMNet.",
                    selector: '.hub-notice, [data-testid="stColumn"]:nth-child(2), [data-testid="column"]:nth-child(2)'
                },
                {
                    title: "3. Xem bảng giá dự báo",
                    text: "Bảng hiển thị giá dự kiến theo 7 mốc cho các mặt hàng đang có dữ liệu, với đơn vị tương ứng trên giao diện.",
                    selector: '[data-testid="stDataFrame"], .table-wrap'
                },
                {
                    title: "4. Đọc biểu đồ so sánh các mốc",
                    text: "Rê chuột lên điểm dữ liệu để xem giá, hoặc bấm tên mốc trong chú giải để ẩn và hiện từng đường dự báo.",
                    selector: '[data-testid="stPlotlyChart"], .stPlotlyChart'
                },
                {
                    title: "5. Xuất file báo cáo & Chuyển bước",
                    text: "Bấm nút 'Xuất Bảng Dự Báo' để tải file Excel/CSV gửi lãnh đạo, hoặc chuyển sang menu 'Đánh giá mô hình' để kiểm tra sai số thực tế.",
                    selector: '[data-testid="stDownloadButton"], [data-testid="stSidebar"], .stDownloadButton'
                }
            ],
            metrics: [
                {
                    title: "1. Đọc chỉ số sai số (MAPE & MAE)",
                    text: "Theo dõi sai số giữa dự báo và giá thị trường. MAPE dưới 7% được hiển thị màu xanh theo ngưỡng cấu hình; trên 10% hệ thống sẽ xem xét tối ưu GUMNet Candidate khi đủ dữ liệu.",
                    selector: '[data-testid="stHorizontalBlock"], [data-testid="column"], [data-testid="stMetric"]'
                },
                {
                    title: "2. MAPE theo mốc dự báo",
                    text: "Bảng này giúp so sánh phần trăm sai lệch ở từng mốc từ 1 đến 60 ngày.",
                    selector: '[id="mape-theo-moc-du-bao"]',
                    groupTargets: [{selector: '[id="mape-theo-moc-du-bao"]'}, {selector: '[data-testid="stDataFrame"]', index: 0}]
                },
                {
                    title: "3. MAPE theo mặt hàng",
                    text: "Xem mặt hàng nào đang có sai số tương đối cao hoặc thấp hơn trong tập dữ liệu đánh giá.",
                    selector: '[id="mape-theo-mat-hang"]',
                    groupTargets: [{selector: '[id="mape-theo-mat-hang"]'}, {selector: '[data-testid="stDataFrame"]', index: 1}]
                },
                {
                    title: "4. Chi tiết sai lệch tuyệt đối MAE",
                    text: "MAE thể hiện độ lệch trung bình theo đơn vị giá. Nên đọc cùng MAPE và số lượng mẫu đánh giá.",
                    selector: '[id="chi-tiet-sai-lech-gia-tuyet-doi-mae"]',
                    groupTargets: [{selector: '[id="chi-tiet-sai-lech-gia-tuyet-doi-mae"]'}, {selector: '[data-testid="stDataFrame"]', index: 2}]
                },
                {
                    title: "5. Xem xu hướng sai số",
                    text: "Biểu đồ cho thấy sai số thay đổi giữa các mốc dự báo để hỗ trợ nhận biết mốc cần theo dõi thêm.",
                    selector: '[data-testid="stPlotlyChart"], .stPlotlyChart'
                },
                {
                    title: "6. Tối ưu mô hình khi cần",
                    text: "Nút này dùng để yêu cầu Finetune thủ công. Chỉ nên dùng khi muốn đánh giá lại model với dữ liệu hiện tại; hệ thống sẽ khóa nút khi một tiến trình đang chạy.",
                    selector: '[data-testid="stButton"] button, [data-testid="stButton"]'
                }
            ],
            history: [
                {
                    title: "1. Xem và xuất nhật ký các đợt nạp",
                    text: "Bảng trên liệt kê các đợt nạp dữ liệu. Nút xuất ngay bên dưới tải báo cáo tổng hợp lịch sử nạp file.",
                    selector: '[data-testid="stDataFrame"]',
                    groupTargets: [{selector: '[data-testid="stDataFrame"]', index: 0}, {selector: '[data-testid="stDownloadButton"]', index: 0}]
                },
                {
                    title: "2. Xem và xuất chi tiết một đợt",
                    text: "Chọn một đợt, xem bảng Thực tế và Dự báo, rồi dùng nút xuất bên cạnh để tải dữ liệu của riêng đợt đang chọn.",
                    selector: '[data-testid="stSelectbox"]',
                    groupTargets: [{selector: '[data-testid="stSelectbox"]', index: 0}, {selector: '[data-testid="stDownloadButton"]', index: 1}, {selector: '[data-testid="stDataFrame"]', index: 1}]
                }
            ],
            charts: [
                {
                    title: "1. Chọn dữ liệu cần quan sát",
                    text: "Dùng các bộ lọc trên trang Biểu đồ để chọn mặt hàng, khoảng dữ liệu hoặc nội dung cần so sánh.",
                    selector: '[data-testid="stSelectbox"], [data-testid="stMultiSelect"], [data-testid="stDateInput"]'
                },
                {
                    title: "2. Đọc và tương tác với biểu đồ",
                    text: "Rê chuột lên điểm dữ liệu để xem giá trị, dùng thanh công cụ để phóng to, thu nhỏ hoặc tải hình biểu đồ.",
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

        function pickGroupedTarget(targetSpecs) {
            const scope = doc.querySelector('[data-testid="stMain"]')
                || doc.querySelector('section.main')
                || doc.querySelector('.main')
                || doc;
            const elements = [];
            for (const spec of (targetSpecs || [])) {
                try {
                    const matches = scope.querySelectorAll(spec.selector);
                    const el = matches[spec.index || 0];
                    if (el && el.getBoundingClientRect().height > 0) elements.push(el);
                } catch (e) { /* bỏ qua selector không hợp lệ */ }
            }
            if (!elements.length) return null;
            return {
                _oilElements: elements,
                getBoundingClientRect: function() {
                    const rects = elements.map(el => el.getBoundingClientRect());
                    const left = Math.min(...rects.map(r => r.left));
                    const top = Math.min(...rects.map(r => r.top));
                    const right = Math.max(...rects.map(r => r.right));
                    const bottom = Math.max(...rects.map(r => r.bottom));
                    return {left, top, right, bottom, width: right - left, height: bottom - top};
                },
                scrollIntoView: function(options) { elements[0].scrollIntoView(options); }
            };
        }

        function waitForTargetElement(step, callback, maxTries = 30, interval = 120) {
            let tries = 0;
            function check() {
                const el = step.groupTargets
                    ? pickGroupedTarget(step.groupTargets)
                    : pickBestElement(step.selector);
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
            const previousTargets = doc.querySelectorAll('.oil-tour-focus');
            previousTargets.forEach(function(prev) {
                prev.classList.remove('oil-tour-focus');
                prev.style.removeProperty('outline-width');
                prev.style.removeProperty('outline-style');
                prev.style.removeProperty('outline-color');
                prev.style.removeProperty('outline-offset');
                prev.style.removeProperty('box-shadow');
            });

            if (idx >= activeTour.length) {
                endTour();
                return;
            }

            const step = activeTour[idx];

            doc.getElementById('oil-tour-title').textContent = step.title;
            doc.getElementById('oil-tour-text').textContent = step.text;
            doc.getElementById('oil-tour-count').textContent = `Bước ${idx + 1} / ${activeTour.length}`;
            doc.getElementById('oil-tour-next').textContent = (idx === activeTour.length - 1) ? 'Hoàn tất ✓' : 'Tiếp theo ➔';

            waitForTargetElement(step, function(targetEl) {
                if (targetEl) {
                    const focusTargets = targetEl._oilElements || [targetEl];
                    focusTargets.forEach(el => el.classList.add('oil-tour-focus'));
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
            const previousTargets = doc.querySelectorAll('.oil-tour-focus');
            previousTargets.forEach(function(prev) {
                prev.classList.remove('oil-tour-focus');
                prev.style.removeProperty('outline-width');
                prev.style.removeProperty('outline-style');
                prev.style.removeProperty('outline-color');
                prev.style.removeProperty('outline-offset');
                prev.style.removeProperty('box-shadow');
            });
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
                charts: "Biểu đồ",
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
    _hw_tour_str = "thời gian tham khảo 10–20 giây/mốc trên GPU NVIDIA CUDA" if _is_gpu else "thời gian tham khảo khoảng 1–2 phút/mốc trên CPU 6 vCPUs"
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
BACKTEST_MIN_DATE = pd.Timestamp("2025-09-19")
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

def preview_uploaded_dates(uploaded_file):
    """Đọc nhanh khoảng ngày (min, max) của 1 file vừa CHỌN (chưa lưu ra đĩa, chưa xử lý gì) —
    dùng để người dùng xem trước file mới hay cũ trước khi bấm nút xử lý thật sự.
    """
    try:
        uploaded_file.seek(0)
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        df.columns = [str(c).strip() for c in df.columns]
        potential_date_cols = [c for c in df.columns if any(x in c.lower() for x in ["ng", "date", "time"])]
        if not potential_date_cols:
            return None, None
        dates = pd.to_datetime(df[potential_date_cols[0]], errors="coerce", format="mixed").dropna()
        if dates.empty:
            return None, None
        return dates.min(), dates.max()
    except Exception:
        return None, None
    finally:
        uploaded_file.seek(0)

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

def load_model(name, horizon):
    prefix = 'gumnet' if name == 'GUMNet' else 'hybrid'
    path = ROOT / 'checkpoints_multi' / f'{prefix}_h{horizon}.pt'
    stamp = path.stat() if path.exists() else None
    version = (stamp.st_mtime_ns, stamp.st_ctime_ns, stamp.st_size) if stamp else None
    return _load_model_version(name, horizon, version)


@st.cache_resource(max_entries=14)
def _load_model_version(name, horizon, version):
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

def run_upload_simulation(base_path, upload_files, start_date, sel_horizons=None, sel_models=None):
    # Lỗi #13 (đã xác nhận): trước đây dùng biến `sel_horizons` mà hàm này không nhận làm tham
    # số và cũng không phải biến global -> NameError mỗi khi có dữ liệu mới cần backtest, bị
    # nuốt bởi except bên dưới khiến toàn bộ tính năng "Đánh giá mô hình" luôn thất bại âm thầm.
    if sel_horizons is None or len(sel_horizons) == 0:
        sel_horizons = HORIZONS
    # Trước đây hàm này luôn lặp qua TOÀN BỘ MODEL_DEFS (cả GUMNet lẫn HybridTriNet) bất kể người
    # dùng đang chọn model nào ở sidebar — khiến việc đối chiếu chạy lâu gấp đôi không cần thiết.
    if sel_models is None or len(sel_models) == 0:
        sel_models = list(MODEL_DEFS.keys())
    base_full = load_df(base_path)
    base = base_full[base_full[DATE_COL] < start_date].copy()
    all_records = []
    
    # Gộp toàn bộ dữ liệu thực tế lịch sử và các file upload để làm nguồn tra cứu giá thực tế
    actual_dfs = [base_full]
    for fp in upload_files:
        df_tmp = load_df(fp)
        if not df_tmp.empty:
            actual_dfs.append(df_tmp)
    # keep="last": file upload (nằm sau base_full trong actual_dfs) phải thắng dữ liệu gốc
    # khi trùng ngày, để tính năng "xác nhận ghi đè ngày cũ" có tác dụng thật.
    full_actuals = pd.concat(actual_dfs, ignore_index=True).drop_duplicates(subset=[DATE_COL], keep="last").sort_values(DATE_COL)
    
    with open(ROOT / "sim_log.txt", "w", encoding="utf-8") as logf:
        logf.write(f"Simulation started. Files: {len(upload_files)}\n")
        
        status_text = st.empty()
        task_idx = 0
        total_tasks = len(upload_files) * len(sel_models)

        
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

                for mname in sel_models:
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
        
    # keep="last": file upload (nối sau base_full) phải thắng dữ liệu gốc khi trùng ngày,
    # để tính năng "xác nhận ghi đè ngày cũ" hiển thị đúng giá đã cập nhật.
    latest_df = latest_df.drop_duplicates(subset=[DATE_COL], keep="last").sort_values(DATE_COL).reset_index(drop=True)
    
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
    /* Ẩn icon nhỏ mặc định trong nút (đã có icon lớn ở trên). Theo yêu cầu: đổi hẳn nút chữ
       "Chọn file dữ liệu" thành 1 icon vuông dấu "+" thay vì nút chữ như trước. */
    section[data-testid="stFileUploaderDropzone"] span[data-testid="stIconMaterial"] {
        display: none !important;
    }
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        min-height: 40px !important;
        padding: 0 !important;
        border-radius: 8px !important;
        position: relative !important;
    }
    /* Thẻ <p> chứa chữ do Streamlit tự render không thật sự rộng bằng cả nút (chỉ vừa khít
       chữ), nên set display:flex/width:100% trên chính nó không đủ để canh giữa — ép nó phủ
       kín toàn bộ nút bằng position:absolute mới canh giữa đúng, không lệch. */
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] p {
        font-size: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 0 !important;
        position: absolute !important;
        inset: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] p::after {
        content: "+";
        font-size: 22px !important;
        font-weight: 800;
        line-height: 1 !important;
        color: #00ad91;
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
    /* Icon "+" ngay cạnh (các) file đã chọn — gợi ý trực quan là vẫn bấm/kéo thêm được file
       khác, tránh người dùng tưởng ô này chỉ chọn được đúng 1 file. Chỉ là icon minh họa
       (::after không bấm được) — thao tác thêm file thật vẫn qua nút/khung phía trên. */
    div[data-testid="stFileChips"]::after {
        content: "+";
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px;
        height: 26px;
        margin-left: 4px;
        border: 1.5px dashed #00ad91;
        border-radius: 50%;
        color: #00ad91;
        font-size: 15px;
        font-weight: 800;
        flex-shrink: 0;
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

CACHE_FILE = ROOT / "simulation_cache.json"

# Khóa job huấn luyện bằng file trên đĩa: đây là app LAN nhiều người cùng truy cập chung
# một server Streamlit, nên khóa phải chặn được cả những phiên/tab khác, không chỉ session
# hiện tại (session_state không đủ vì mỗi tab/trình duyệt có session_state riêng).
TRAIN_LOCK_FILE = ROOT / ".training.lock"

def _pid_alive(pid):
    try:
        return process_alive(pid)
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
        # Nếu pid chưa được gán (vừa tạo để giữ chỗ trước khi Popen)
        if pid is None and info.get("job_id"):
            started = pd.to_datetime(info.get("started_at"), errors="coerce")
            if pd.notna(started) and (pd.Timestamp.now() - started).total_seconds() < 30:
                return info
    except Exception:
        pass
    # Lock rác (job cũ bị crash/kill mà không dọn được) -> xoá để không khoá cứng vĩnh viễn
    try:
        TRAIN_LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return None

def acquire_training_lock(models, horizons, job_id=None, pid=None):
    """
    Tạo khóa huấn luyện nguyên tử (mode='x') hoặc cập nhật PID cho job_id hiện tại.
    Trả về job_id.
    """
    active = get_active_training_lock()
    if active:
        # Nếu cùng job_id thì cho phép cập nhật PID
        if job_id and active.get("job_id") == job_id:
            active["pid"] = pid or active.get("pid")
            active["models"] = models
            active["horizons"] = horizons
            tmp = TRAIN_LOCK_FILE.with_suffix(".lock.tmp")
            tmp.write_text(json.dumps(active, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, TRAIN_LOCK_FILE)
            return job_id
        raise RuntimeError(f"Một tiến trình huấn luyện khác (Job ID: {active.get('job_id')}) đang chạy.")

    actual_job_id = job_id or uuid.uuid4().hex[:12]
    data = {
        "job_id": actual_job_id,
        "pid": pid,
        "models": models,
        "horizons": horizons,
        "started_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    content = json.dumps(data, ensure_ascii=False, indent=2)

    try:
        with open(TRAIN_LOCK_FILE, mode="x", encoding="utf-8") as f:
            f.write(content)
    except FileExistsError:
        active = get_active_training_lock()
        if active and active.get("job_id") != actual_job_id:
            raise RuntimeError(f"Một tiến trình huấn luyện khác (Job ID: {active.get('job_id')}) vừa chiếm quyền chạy.")
        tmp = TRAIN_LOCK_FILE.with_suffix(".lock.tmp")
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, TRAIN_LOCK_FILE)

    return actual_job_id

def release_training_lock(job_id=None):
    if not TRAIN_LOCK_FILE.exists():
        return
    try:
        info = json.loads(TRAIN_LOCK_FILE.read_text(encoding="utf-8"))
        if job_id is not None and info.get("job_id") != job_id:
            # Không được xóa lock của job khác!
            return
        TRAIN_LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass

CKPT_BACKUP_DIR = ROOT / "checkpoints_backup"
CKPT_BACKUP_KEEP = 5  # chỉ giữ lại N bản sao lưu gần nhất, tự dọn bản cũ hơn để không phình đĩa

def backup_checkpoints_before_training(job_id, models, horizons):
    """Sao lưu đúng các checkpoint (+ metadata Hybrid) SẮP bị job này ghi đè, trước khi huấn
    luyện — để có thể khôi phục nếu model mới ra kết quả tệ hơn. Không sao lưu toàn bộ thư mục
    (tốn chỗ, chậm) — chỉ sao lưu đúng phần liên quan tới lựa chọn hiện tại.
    """
    import shutil
    dest = CKPT_BACKUP_DIR / job_id
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for h in horizons:
        for m in models:
            prefix = "gumnet" if m == "GUMNet" else "hybrid"
            ckpt_f = ROOT / "checkpoints_multi" / f"{prefix}_h{h}.pt"
            if ckpt_f.exists():
                shutil.copy2(ckpt_f, dest / ckpt_f.name)
                saved.append(ckpt_f.name)
            if m != "GUMNet":
                meta_dir = ROOT / "checkpoints_multi" / f"hybrid_h{h}_meta"
                if meta_dir.exists():
                    shutil.copytree(meta_dir, dest / meta_dir.name, dirs_exist_ok=True)
    # Dọn bớt bản sao lưu cũ, chỉ giữ CKPT_BACKUP_KEEP job gần nhất
    try:
        all_backups = sorted(CKPT_BACKUP_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in all_backups[CKPT_BACKUP_KEEP:]:
            shutil.rmtree(old, ignore_errors=True)
    except Exception:
        pass
    return saved

def list_checkpoint_backups():
    """Trả về danh sách các bản sao lưu, mới nhất trước, kèm thời điểm tạo."""
    if not CKPT_BACKUP_DIR.exists():
        return []
    backups = []
    for d in CKPT_BACKUP_DIR.iterdir():
        if d.is_dir():
            backups.append({"job_id": d.name, "mtime": d.stat().st_mtime,
                             "files": [f.name for f in d.iterdir()]})
    return sorted(backups, key=lambda b: b["mtime"], reverse=True)

def restore_checkpoint_backup(job_id):
    """Khôi phục lại đúng các file đã sao lưu của 1 job — dùng khi model mới huấn luyện tệ hơn."""
    import shutil
    src = CKPT_BACKUP_DIR / job_id
    if not src.exists():
        raise FileNotFoundError(f"Không tìm thấy bản sao lưu cho job {job_id}")
    restored = []
    for item in src.iterdir():
        if item.is_dir():
            dst = ROOT / "checkpoints_multi" / item.name
            shutil.copytree(item, dst, dirs_exist_ok=True)
            restored.append(item.name)
        else:
            dst = ROOT / "checkpoints_multi" / item.name
            shutil.copy2(item, dst)
            restored.append(item.name)
    return restored

VAL_LOSS_RE = re.compile(r"(?:Best Val Loss:|best_val=)\s*([\d.]+)")
HZ_START_RE = re.compile(r"ĐANG HUẤN LUYỆN MỐC:\s*(\d+)\s*NGÀY")

def _process_log_line(line, hz_status, current_hz, hz_completed, total_hz):
    """
    Phân tích một dòng log để cập nhật trạng thái các mốc horizon và tiến độ.
    Trả về (new_current_hz, new_hz_completed, changed_flag, notify_msg)
    """
    hz_match = HZ_START_RE.search(line)
    loss_match = VAL_LOSS_RE.search(line)
    changed = False
    notify_msg = None

    if hz_match:
        h_started = int(hz_match.group(1))
        if h_started in hz_status:
            current_hz = h_started
            if hz_status[h_started]["state"] != "running" and hz_status[h_started]["state"] != "done":
                hz_status[h_started]["state"] = "running"
                changed = True
        notify_msg = f"📌 {line.strip()}"
    elif loss_match and current_hz is not None and current_hz in hz_status:
        val = float(loss_match.group(1))
        if hz_status[current_hz]["state"] != "done":
            hz_status[current_hz]["state"] = "done"
            hz_status[current_hz]["val_loss"] = val
            hz_completed = min(hz_completed + 1, total_hz)
            changed = True
        elif hz_status[current_hz]["val_loss"] is None:
            hz_status[current_hz]["val_loss"] = val
            changed = True

    return current_hz, hz_completed, changed, notify_msg


def get_dir_fingerprint():
    data_dir = ROOT / "datasets"
    files = [f for f in data_dir.glob("*") if f.suffix.lower() in [".xlsx", ".xls", ".csv"] and not f.name.startswith("~$")]
    return dataset_fingerprint(data_dir, HORIZONS)

fingerprint = get_dir_fingerprint()

@st.cache_data
def get_sorted_files(fp):
    data_dir = ROOT / "datasets"
    files = [f for f in data_dir.glob("*") if f.suffix.lower() in [".xlsx", ".xls", ".csv"]]
    info = []
    for f in files:
        df = load_df(f)
        # Lỗi thật đã bắt được: 1 file lỡ lọt vào datasets/ (thiếu cột Ngày hợp lệ) làm SẬP TOÀN
        # BỘ APP mỗi lần tải trang (KeyError 'Ngày' ở đây), vì trước chỉ kiểm tra .empty chứ
        # không kiểm tra cột Ngày có tồn tại không. Giờ bỏ qua an toàn file nào thiếu cột Ngày,
        # không để 1 file rác làm treo cả ứng dụng cho mọi người dùng.
        if not df.empty and DATE_COL in df.columns:
            info.append({"path": str(f), "max_date": df[DATE_COL].max(), "name": f.name, "rows": len(df)})
    info.sort(key=lambda x: x["max_date"])
    return info

file_info = get_sorted_files(fingerprint)
file_paths = [Path(i["path"]) for i in file_info]

# Use the complete accepted dataset when deciding whether an uploaded row is
# new or modified. Comparing only with the built-in CSV causes an accepted
# correction to be reported again on every subsequent upload.
_existing_frames = [base_full_orig]
for _existing_path in file_paths:
    _existing_df = load_df(_existing_path)
    if not _existing_df.empty and DATE_COL in _existing_df.columns:
        _existing_frames.append(_existing_df)
existing_records_full = (
    pd.concat(_existing_frames, ignore_index=True)
    .drop_duplicates(subset=[DATE_COL], keep="last")
    .sort_values(DATE_COL)
    .reset_index(drop=True)
)

# CUTOFF_DATE: chỉ lấy các điểm backtest trong 365 ngày gần nhất TÍNH THEO NGÀY MỚI NHẤT
# đang có trong dữ liệu (dataset gốc + các file đã upload) — tự động trôi theo dữ liệu mới,
# không còn là một ngày cố định phải nhớ sửa tay mỗi năm.
_known_max_dates = [base_full_orig[DATE_COL].max()] + [i["max_date"] for i in file_info]
_known_max_dates = [d for d in _known_max_dates if pd.notna(d)]
_latest_known_date = max(_known_max_dates) if _known_max_dates else pd.Timestamp.now()
CUTOFF_DATE = max(BACKTEST_MIN_DATE, _latest_known_date - pd.Timedelta(days=365))

# Load cache
cache_mismatch = False
if CACHE_FILE.exists():
    try:
        cached_fp, combined = read_cache(CACHE_FILE)
        if cached_fp != fingerprint:
            cache_mismatch = True
    except Exception as e:
        st.warning(f"Không đọc được kết quả đối chiếu đã lưu: {e}")
        combined = None
else:
    combined = None

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
    "📈  Biểu đồ",
    "❓  Hướng dẫn sử dụng"
]
# Không được gán st.session_state["main_nav_radio"] SAU KHI widget cùng key này đã được tạo ở
# một lượt chạy trước (Streamlit ném StreamlitAPIException) — các nút "chuyển trang" trong app chỉ
# đặt cờ tạm _pending_nav rồi st.rerun(); ở đầu lượt chạy MỚI này (trước khi radio được tạo bên
# dưới), lấy cờ đó ra để cập nhật main_nav_radio thì mới hợp lệ.
if "_pending_nav" in st.session_state:
    st.session_state["main_nav_radio"] = st.session_state.pop("_pending_nav")

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
    <b style="color:#075f57; font-size:13.5px;">🧠 GUMNet Production</b>
    <div style="color:#116d5c; font-size:11px; margin-top:3px;">Đã nạp checkpoint cho 7/7 mốc thời gian</div>
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
        <small style="margin-left:14px; color:#116d5c;">Có hỗ trợ tăng tốc phần cứng</small>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="padding:11px 13px; border-radius:9px; background:#fff7e7; border:1px solid #f6d58c; color:#794800; font-size:12px;">
        <span class="pulsing-dot-cpu"></span>
        <b>Server: CPU Doanh Nghiệp</b><br>
        <small style="margin-left:14px; color:#794800;">6 vCPUs · Thời gian dự báo tham khảo &lt; 1 giây</small>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 2. ĐỒNG BỘ DỮ LIỆU ĐÁNH GIÁ (BACKTESTING) — JOB NỀN ĐỘC LẬP, KHÔNG PHỤ THUỘC SESSION
# ==========================================
# Trước đây (2 lần sửa liên tiếp): chạy đồng bộ NGAY TRONG script Streamlit, chặn cả trang bằng
# st.spinner() — vừa chặn người dùng chờ, vừa gắn với session hiện tại (đổi trang/đóng tab giữa
# chừng thì mất luôn tiến độ, F5 lại là tính lại từ đầu). Giờ chuyển hẳn sang TIẾN TRÌNH NỀN ĐỘC
# LẬP (run_backtest_job.py, xem file đó) — không phải Thread trong session, mà 1 process hệ điều
# hành thật, chạy tới khi xong bất kể Streamlit đang làm gì, đổi trang/rerun không ảnh hưởng.
# Trạng thái đọc từ 2 file trên đĩa (.backtest.lock, .backtest_job.json) — không dùng
# session_state làm nguồn thật (session_state chỉ là bộ nhớ tạm của riêng 1 tab/session).
BACKTEST_LOCK_FILE = ROOT / ".backtest.lock"
BACKTEST_STATUS_FILE = ROOT / ".backtest_job.json"

def _replace_with_retry(tmp, dest, attempts=6, delay=0.05):
    """os.replace() trên Windows có thể ném PermissionError [WinError 32] nếu file đích đang bị
    tiến trình khác mở đúng lúc đó (VD: nhiều lượt rerun của Streamlit cùng ghi status gần như
    đồng thời). Đây chỉ là tranh chấp thoáng qua, không phải lỗi thật — thử lại vài lần cách nhau
    vài chục mili-giây gần như luôn tự qua được, không cần người dùng thấy lỗi đỏ."""
    for i in range(attempts):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)

def get_backtest_status():
    if not BACKTEST_STATUS_FILE.exists():
        return None
    try:
        return json.loads(BACKTEST_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None

def _backtest_job_alive(status):
    """status pending/running CHƯA CHẮC job còn thật sự sống (có thể bị kill/crash không kịp ghi
    'failed') — đối chiếu thêm với .backtest.lock (có PID) để chắc chắn, giống hệt cách app đã
    làm với khóa huấn luyện."""
    if not status or status.get("status") not in ("pending", "running"):
        return False
    if not BACKTEST_LOCK_FILE.exists():
        return False
    try:
        lock_info = json.loads(BACKTEST_LOCK_FILE.read_text(encoding="utf-8"))
        return lock_info.get("job_id") == status.get("job_id") and _pid_alive(lock_info.get("pid"))
    except Exception:
        return False

def spawn_backtest_job(fp, models, files, cutoff_date):
    job_id = uuid.uuid4().hex[:12]
    tmp = BACKTEST_STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "job_id": job_id, "fingerprint": fp, "status": "pending",
        "started_at": None, "finished_at": None, "error": None,
    }, ensure_ascii=False), encoding="utf-8")
    _replace_with_retry(tmp, BACKTEST_STATUS_FILE)  # ghi "pending" TRƯỚC khi Popen, để lượt render kế
    # tiếp (dù rất sát ngay sau) đã thấy có job đang chờ, tránh 2 lượt rerun gần nhau cùng spawn.
    cmd = [
        sys.executable, str(ROOT / "run_backtest_job.py"),
        "--job-id", job_id, "--fingerprint", fp,
        "--cutoff-date", cutoff_date.strftime("%Y-%m-%d"),
        "--models"] + models + ["--files"] + [str(p) for p in files]
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)
    return job_id

def ensure_backtest_job_running(fp, models, files, cutoff_date, force=False):
    """Gọi ở MỌI trang, MỌI lượt rerun (không riêng gì trang Đánh giá) — tự đảm bảo luôn có
    đúng 1 job đối chiếu chạy nền cho đúng dữ liệu MỚI NHẤT, không cần người dùng bấm gì.
    """
    if not files:
        return
    # Lỗi kiến trúc đã phát hiện: hệ thống đối chiếu cũ này (spawn_backtest_job) và pipeline tự
    # động huấn luyện mới (pipeline_engine) cùng ghi vào chung 1 file simulation_cache.json và
    # cùng chạy model trên chung 1 GPU — nếu cả 2 cùng chạy một lúc sẽ tranh nhau và có thể ghi
    # đè kết quả của nhau. Trong lúc pipeline mới đang chạy thật, hệ thống cũ tạm nhường, không
    # tự tạo job riêng — tránh chạy chồng chéo lãng phí và tranh chấp file/GPU.
    if pipeline_engine.get_pipeline_status().get("is_running") or pipeline_engine.pipeline_lock_active():
        return
    status = get_backtest_status()
    alive = _backtest_job_alive(status)
    if not force:
        if status and status.get("fingerprint") == fp and alive:
            return  # đúng job cho đúng dữ liệu này đang chạy rồi -> không tạo trùng
        if status and status.get("fingerprint") == fp and status.get("status") == "success" and CACHE_FILE.exists() and not cache_mismatch and not combined.empty:
            return  # đã có kết quả đúng dữ liệu này rồi -> khỏi chạy lại
    if alive and status.get("fingerprint") != fp:
        # Có job KHÁC (dữ liệu cũ hơn) đang chạy dở — để nó chạy nốt, KHÔNG chen ngang (chỉ 1
        # job/thời điểm). Khi nó xong, lượt rerun kế tiếp (của bất kỳ ai, bất kỳ trang nào) sẽ
        # tự thấy fingerprint hiện tại vẫn khác "success" và tự tạo job mới cho dữ liệu mới nhất
        # — đây chính là cơ chế "chạy nối tiếp theo dữ liệu mới nhất" mà không cần vòng lặp riêng.
        return
    spawn_backtest_job(fp, models, files, cutoff_date)

# The upload pipeline is the single automatic owner of backtest/cache updates.
# Keep the legacy job helpers only for compatibility with old status files; do
# not start a second model process on every Streamlit rerun.
_bt_status = get_backtest_status()
_backtest_running_now = _bt_status is not None and _bt_status.get("fingerprint") == fingerprint and _backtest_job_alive(_bt_status)
_backtest_failed_current = _bt_status is not None and _bt_status.get("fingerprint") == fingerprint and _bt_status.get("status") == "failed"
_backtest_stale = (cache_mismatch or (combined.empty and file_paths)) and not _backtest_running_now

if not combined.empty:
    if "Mặt hàng" in combined.columns:
        combined = combined.rename(columns={"Mặt hàng": "Target"})
    df_view = combined[(combined["Model"].isin(sel_models)) & (combined[DATE_COL] >= CUTOFF_DATE)]
else:
    df_view = pd.DataFrame()


_LAUNCH_REFUSE_MESSAGES = {
    'already_trained': 'Dữ liệu này đã được huấn luyện và đánh giá. Hãy cập nhật dữ liệu mới trước lượt tối ưu tiếp theo.',
    "pipeline_running": "⏳ Một đợt xử lý khác đang chạy. Vui lòng đợi đợt đó xong rồi thử lại.",
    "backtest_running": "⏳ Hệ thống đang chạy đối chiếu nền cho dữ liệu hiện tại. Đợi xong rồi thử lại để tránh 2 tiến trình tranh nhau.",
    "pipeline_locked": "⏳ Hệ thống đang bận giữ khóa xử lý. Vui lòng thử lại sau giây lát.",
    "start_failed": "❌ Không khởi động được tiến trình nền. Xem lại nhật ký hệ thống.",
}


def launch_pipeline_with_feedback(batch_id, new_rows, file_names, force_retrain=False):
    """Khởi động pipeline và BÁO RÕ cho người dùng khi bị từ chối.

    Trước đây các nút gọi launch_pipeline_background() rồi st.rerun() ngay mà không xem
    kết quả — bị từ chối thì người dùng không thấy gì cả, tưởng bấm hụt. Chỉ rerun khi
    thật sự khởi động được; bị từ chối thì giữ nguyên trang để đọc được thông báo.
    """
    result = pipeline_engine.launch_pipeline_background(
        batch_id, new_rows, file_names, force_retrain=force_retrain
    )
    if result.get("started"):
        st.rerun()
    else:
        reason = result.get("reason", "")
        st.warning(_LAUNCH_REFUSE_MESSAGES.get(reason, f"Chưa khởi động được tiến trình nền ({reason})."))
    return result


# ==========================================
# 3. NỘI DUNG TỪNG TRANG (THEO MẪU ENTERPRISE MOCKUP)
# ==========================================

def render_global_pipeline_banner():
    status_data = pipeline_engine.get_pipeline_status()
    st_val = status_data.get("status")
    is_running = status_data.get("is_running", False)

    if st_val == "idle" and not is_running:
        return False

    steps = status_data.get("steps", [])
    step_title = status_data.get("step_title", "Cập nhật dữ liệu")
    details = status_data.get("details", "")
    updated_at = status_data.get("updated_at", "")

    step_html_items = []
    for s in steps:
        state = s.get("state", "waiting")
        title = s.get("title", "")
        if state == "done":
            step_html_items.append(
                f'<span style="color:#087762; font-weight:700; font-size:12.5px; display:inline-flex; align-items:center; gap:4px;">'
                f'<span style="color:#00ad91;">✓</span> {title}</span>'
            )
        elif state == "running":
            step_html_items.append(
                f'<span style="color:#075f57; font-weight:800; font-size:12.5px; display:inline-flex; align-items:center; gap:4px; background:#dff5f1; padding:2px 8px; border-radius:6px;">'
                f'<span style="color:#00ad91; animation:pulseDot 1.5s infinite;">●</span> {title}</span>'
            )
        elif state == "failed":
            step_html_items.append(
                f'<span style="color:#dc2626; font-weight:700; font-size:12.5px; display:inline-flex; align-items:center; gap:4px;">'
                f'✕ {title}</span>'
            )
        else:
            step_html_items.append(
                f'<span style="color:#94a3b8; font-weight:500; font-size:12.5px; display:inline-flex; align-items:center; gap:4px;">'
                f'○ {title}</span>'
            )

    steps_row = " &nbsp; <span style='color:#cbd5e1;'>➔</span> &nbsp; ".join(step_html_items)

    banner_bg = "#f0fdf4" if st_val == "complete" else ("#fff1f2" if st_val == "failed" else "#f0fdfa")
    banner_border = "#bbf7d0" if st_val == "complete" else ("#fecdd3" if st_val == "failed" else "#99f6e4")
    border_left_color = "#16a34a" if st_val == "complete" else ("#e11d48" if st_val == "failed" else "#0d9488")

    if st_val == "failed":
        error_detail = html.escape(str(status_data.get("error") or "Tiến trình gặp sự cố hoặc bị gián đoạn."))
        tip_text = f"⚠️ {error_detail} Bấm nút bên dưới để đóng và đặt lại ban đầu."
        tip_color = "#b91c1c"
        tip_bg = "#fee2e2"
    elif st_val == "complete":
        tip_text = "✅ Đợt xử lý đã hoàn tất thành công. Bạn có thể xem kết quả hoặc đóng thông báo."
        tip_color = "#047857"
        tip_bg = "#d1fae5"
    else:
        tip_text = "💡 Trang sẽ tự mở lại khi xử lý xong, không cần bấm gì thêm."
        tip_color = "#087762"
        tip_bg = "rgba(0,173,145,0.08)"

    st.markdown(f"""
    <div style="background:{banner_bg}; border:1px solid {banner_border}; border-left:5px solid {border_left_color}; border-radius:10px; padding:12px 18px; margin-bottom:12px; box-shadow:0 2px 6px rgba(0,0,0,0.03);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:8px;">
            <div style="font-size:13px; font-weight:800; color:#0f172a; text-transform:uppercase; letter-spacing:0.04em;">
                TIẾN TRÌNH HỆ THỐNG: <span style="color:{border_left_color};">{step_title}</span>
            </div>
            <div style="font-size:11.5px; color:#64748b;">Cập nhật lúc: <b>{updated_at}</b></div>
        </div>
        <div style="padding:4px 0; display:flex; flex-wrap:wrap; align-items:center; gap:6px;">
            {steps_row}
        </div>
        <div style="margin-top:8px; font-size:12.5px; color:#334155; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <span>{details}</span>
            <span style="color:{tip_color}; font-weight:600; font-size:12px; background:{tip_bg}; padding:3px 8px; border-radius:6px;">{tip_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Màn hình chờ khóa thao tác chuột khi tiến trình đang chạy (tự động mở khóa khi hoàn tất)
    if is_running:
        st.markdown(f"""
        <style>
        .pipeline-lock-overlay {{
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(15, 23, 42, 0.65);
            backdrop-filter: blur(4px);
            z-index: 999999;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            pointer-events: all;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        .pipeline-lock-box {{
            background: #ffffff;
            color: #0f172a;
            border-radius: 16px;
            padding: 28px 36px;
            max-width: 560px;
            width: 90%;
            box-shadow: 0 25px 50px -12px rgba(0,0,0,0.35);
            text-align: center;
            border: 1px solid #e2e8f0;
        }}
        .pipeline-spinner {{
            border: 4px solid #f1f5f9;
            border-top: 4px solid #00ad91;
            border-radius: 50%;
            width: 44px; height: 44px;
            animation: spinLock 1s linear infinite;
            margin: 0 auto 16px;
        }}
        @keyframes spinLock {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        <div class="pipeline-lock-overlay">
            <div class="pipeline-lock-box">
                <div class="pipeline-spinner"></div>
                <div style="font-size:11.5px; font-weight:800; color:#00ad91; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">
                    HỆ THỐNG ĐANG TỰ ĐỘNG XỬ LÝ DỮ LIỆU
                </div>
                <h3 style="margin:4px 0 10px; font-size:20px; font-weight:800; color:#0f172a;">
                    {step_title}
                </h3>
                <div style="font-size:13.5px; color:#475569; margin-bottom:16px; line-height:1.5;">
                    {details}
                </div>
                <div style="background:#f8fafc; border-radius:8px; padding:10px 14px; font-size:12px; color:#64748b;">
                    🔒 <i>Màn hình đang tạm khóa để bảo vệ dữ liệu. Hệ thống sẽ <b>tự động mở khóa</b> ngay khi hoàn tất.</i>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Nút đóng hoặc đặt lại trạng thái khi tiến trình không còn chạy thật
    if (st_val in ("failed", "complete") or not is_running) and st_val != "idle":
        retrain_dec = status_data.get("retrain_decision") or {}
        can_manual_retrain = (st_val == "complete" and not retrain_dec.get("needed", False)
                              and not pipeline_engine.already_trained(fingerprint))

        if can_manual_retrain:
            _col_space, _col_retrain, _col_dismiss = st.columns([2.2, 1.4, 1.2])
            with _col_retrain:
                if st.button("⚡ Vẫn muốn huấn luyện lại", key="btn_force_retrain_banner",
                             type="secondary", use_container_width=True,
                             help="Kích hoạt quy trình tối ưu GUMNet Candidate ngầm ngay cả khi MAPE đang tốt"):
                    fps = [p.name for p in file_paths]
                    launch_pipeline_with_feedback("FORCE-RETRAIN", 0, fps, force_retrain=True)
            with _col_dismiss:
                if st.button("✕ Đóng thông báo", key="btn_dismiss_pipeline_banner", use_container_width=True):
                    pipeline_engine.reset_pipeline_status()
                    st.rerun()
        else:
            _col_space, _col_btn = st.columns([3.2, 1.3])
            with _col_btn:
                btn_label = "✕ Đóng thông báo & Đặt lại" if st_val == "failed" else "✕ Đóng thông báo"
                if st.button(btn_label, key="btn_dismiss_pipeline_banner", use_container_width=True):
                    pipeline_engine.reset_pipeline_status()
                    st.rerun()

    return bool(is_running)

# Hiển thị thanh tiến trình toàn cục trên mọi trang
is_pipeline_busy = bool(render_global_pipeline_banner())

# Theo yêu cầu: khi pipeline đang xử lý thật (is_pipeline_busy/is_running), khóa toàn trang —
# không hiện nội dung/menu bên dưới, không cho thao tác đi chỗ khác — chỉ hiện đúng 1 màn hình
# tiến trình ở giữa, tự làm mới tới khi thật sự "Hoàn tất" (is_running mới chuyển False).
if is_pipeline_busy:
    time.sleep(2)
    st.rerun()

# ──────────────────────────────────────────
# TRANG 1: DỰ BÁO
# ──────────────────────────────────────────
if nav_choice == "◈  Dự báo":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="color:#00ad91; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.09em;">TRUNG TÂM DỰ BÁO GIÁ DẦU</div>
        <h1 style="margin:4px 0; font-size:28px; font-weight:800; letter-spacing:-.03em;">Dự Báo Thị Trường</h1>
        <p style="margin:0; color:#64748b; font-size:14px;">Tạo dự báo giá đa mốc thời gian từ dữ liệu thị trường mới nhất bằng GUMNet Production.</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_status = st.columns([1.25, 0.75])
    
    with col_input:
        card_input = st.container(border=True)
    with card_input:
        st.markdown("#### Cập nhật dữ liệu thị trường")
        st.caption("Tải tệp Excel (.xlsx, .xls) hoặc CSV chứa cột Ngày và giá thị trường. Hỗ trợ chọn nhiều file cùng lúc:")
        ups = st.file_uploader(
            "Upload file", type=["xlsx", "xls", "csv"], key="uploader_main",
            label_visibility="collapsed", accept_multiple_files=True,
        )
        # Gợi ý "vẫn thêm được file khác" giờ chỉ còn icon dấu "+" cạnh file (CSS, xem
        # stFileChips::after) — bỏ dòng chữ theo yêu cầu, icon đã đủ rõ.
        active_file_paths = list(file_paths)

        if ups:
            st.caption(f"📋 **Kiểm tra xem trước {len(ups)} file đã chọn (độc lập từng file):**")
            previews = data_pipeline.preview_uploaded_files(
                ups, _latest_known_date, existing_records=existing_records_full
            )
            all_valid_count = 0
            confirmed_overwrite_files = set()

            for idx, p in enumerate(previews):
                fname = p["filename"]
                if not p["is_valid"]:
                    st.markdown(f"""
                    <div style="background:#fff1f2; border:1px solid #fecdd3; border-radius:8px; padding:7px 12px; margin-bottom:6px; font-size:12.5px;">
                        <b style="color:#e11d48;">✕ {fname}</b> — <span style="color:#475569;">{p.get('error')}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    all_valid_count += 1
                    d_min_str = p['min_date'].strftime('%d/%m/%Y') if p.get('min_date') else ""
                    d_max_str = p['max_date'].strftime('%d/%m/%Y') if p.get('max_date') else ""
                    
                    if p.get("has_new_data"):
                        st.markdown(f"""
                        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:7px 12px; margin-bottom:6px; font-size:12.5px;">
                            <b style="color:#16a34a;">✓ {fname}</b> — <span style="color:#1e293b;">{p['rows']:,} dòng ({d_min_str} → {d_max_str})</span> · 
                            <b style="color:#059669;">Phát hiện {p.get('new_rows_count', 0)} ngày mới</b>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:7px 12px; margin-bottom:6px; font-size:12.5px;">
                            <b style="color:#0284c7;">ℹ {fname}</b> — <span style="color:#64748b;">Hợp lệ ({p['rows']:,} dòng, đến {d_max_str}) · <b>Dữ liệu đã có sẵn trong hệ thống</b></span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    if p.get("modified_rows_count", 0) > 0:
                        st.markdown(f"""
                        <div style="background:#fffbeb; border:1px solid #fde68a; border-radius:8px; padding:5px 12px; margin-bottom:6px; font-size:12px; color:#b45309;">
                            ⚠️ Phát hiện {p['modified_rows_count']} ngày có giá điều chỉnh so với dữ liệu cũ.
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander(f"Xem chi tiết & xác nhận ghi đè — {fname}"):
                            details = p.get("modified_details", [])
                            if details:
                                detail_df = pd.DataFrame([{
                                    "Ngày": d["date"].strftime("%d/%m/%Y") if hasattr(d["date"], "strftime") else str(d["date"]),
                                    "Mặt hàng": d["column"],
                                    "Giá cũ": d["old_value"],
                                    "Giá mới": d["new_value"],
                                } for d in details])
                                st.dataframe(detail_df, use_container_width=True, hide_index=True)
                            # Lỗi đã sửa: trước đây key chỉ dựa vào sha256 -> nếu 2 file trong
                            # cùng lượt tải có NỘI DUNG GIỐNG HỆT NHAU (kể cả khác tên), key trùng
                            # nhau -> StreamlitDuplicateElementKey crash cả trang. Thêm số thứ tự
                            # (idx) của file trong danh sách để luôn duy nhất, đồng thời giữ sha256
                            # kèm theo để file đứng nguyên vị trí vẫn giữ được trạng thái đã tick
                            # qua các lượt rerun (không tự ý reset checkbox vô cớ).
                            confirm_key = f"confirm_overwrite_{idx}_{p.get('sha256', fname)}"
                            confirm = st.checkbox(
                                f"Tôi xác nhận muốn ghi đè {p['modified_rows_count']} ngày ở trên bằng giá trị mới từ file này",
                                key=confirm_key,
                            )
                            if confirm:
                                # Lỗi đã sửa: trước đây ghi nhận theo TÊN FILE (fname) — nếu 2 file
                                # trùng tên nhưng khác nội dung, tick xác nhận 1 file sẽ vô tình
                                # làm cả file kia (chưa tick) cũng bị coi là "đã xác nhận". Dùng
                                # sha256 (nội dung thật) thay vì tên file để phân biệt chính xác
                                # từng file, kể cả khi trùng tên.
                                confirmed_overwrite_files.add(p.get("sha256", fname))
                                st.caption("✅ Các ngày trên sẽ được cập nhật khi bạn bấm nút xử lý bên dưới.")
                            else:
                                st.caption("Chưa xác nhận — các ngày này sẽ tiếp tục giữ nguyên giá trị cũ.")

            files_to_process = [
                p for p in previews
                if p["is_valid"] and (p.get("has_new_data") or p.get("sha256", p["filename"]) in confirmed_overwrite_files)
            ]

            _col_l, _col_mid, _col_r = st.columns([1, 2, 1])
            with _col_mid:
                if len(files_to_process) > 0:
                    btn_text = f"⚡ Kiểm tra & cập nhật {len(files_to_process)} file hợp lệ"
                    pipeline_busy = bool(pipeline_engine.get_pipeline_status().get("is_running"))
                    if pipeline_busy:
                        st.caption("Hệ thống đang xử lý đợt trước. Nút sẽ tự mở lại khi hoàn tất.")
                    if st.button(btn_text, key="btn_process_uploads", type="primary",
                                 use_container_width=True, disabled=pipeline_busy):
                        raw_map = {f.name: f for f in ups}
                        res = data_pipeline.commit_valid_files(
                            previews, raw_map, overwrite_confirmed_files=confirmed_overwrite_files
                        )
                        if res["success"]:
                            file_names = [f["saved_as"] for f in res["saved_files"]]
                            launch = pipeline_engine.launch_pipeline_background(
                                res["batch_id"], res["total_new_rows"], file_names
                            )
                            msg = f"✅ Đã tiếp nhận và cập nhật {len(res['saved_files'])} file ({res['total_new_rows']} ngày mới"
                            if res.get("total_overwritten_rows", 0) > 0:
                                msg += f", {res['total_overwritten_rows']} ngày được ghi đè giá"
                            msg += ")."
                            if launch.get("started"):
                                st.success(msg + " Hệ thống đang kích hoạt đối chiếu ngầm!")
                            else:
                                st.warning("Dữ liệu đã được lưu nhưng tiến trình nền chưa thể khởi động. Hãy đợi tác vụ hiện tại hoàn tất rồi thử cập nhật lại.")
                            st.rerun()
                        else:
                            st.error("Không thể ghi nhận các file đã chọn. Vui lòng kiểm tra lại.")
                elif all_valid_count > 0:
                    st.button("ℹ️ Dữ liệu đã tồn tại (Không có ngày mới)", key="btn_no_new_data", disabled=True, use_container_width=True)
                else:
                    st.button("✕ Tất cả file không hợp lệ", key="btn_all_invalid", disabled=True, use_container_width=True)

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
                Dự báo và huấn luyện đang sử dụng tăng tốc phần cứng khi khả dụng.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="hub-notice cpu">
                <b style="display:block; margin-bottom:2px;">🔵 Đang chạy bằng CPU Doanh Nghiệp</b>
                Dự báo nhanh (&lt; 1s) hoạt động bình thường. Hệ thống tự đối chiếu và tối ưu khi cần.
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
                <b style="font-size:14px; color:#087762;">{'Sẵn sàng' if sum((CKPT_DIR / f'gumnet_h{h}.pt').exists() for h in HORIZONS) == len(HORIZONS) else 'Chưa đầy đủ'} ({sum((CKPT_DIR / f'gumnet_h{h}.pt').exists() for h in HORIZONS)}/{len(HORIZONS)} mốc)</b>
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
    
    ckpts_exist = all((CKPT_DIR / f"gumnet_h{h}.pt").exists() for h in HORIZONS)
    if ckpts_exist:
        show_live_forecasts(base_full_orig, active_file_paths, sel_models, sel_hz_view)
    else:
        st.info("ℹ️ **GUMNet chưa sẵn sàng.** Vui lòng liên hệ người vận hành hệ thống để kiểm tra model.")


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

    # Bỏ nút "Cập nhật (dự phòng)/Thử lại" theo yêu cầu — hệ thống giờ chạy tự động hoàn toàn,
    # không cần nút tay dự phòng nữa. Chỉ còn hiện đúng trạng thái để người dùng biết đang ở đâu.
    if _backtest_running_now:
        st.info("⏳ **Đang tự động cập nhật kết quả đối chiếu...** Trang sẽ tự hiện kết quả mới khi xong, không cần bấm gì — có thể chuyển sang trang khác trong lúc chờ.")
        # Streamlit không tự vẽ lại trang khi 1 tiến trình NỀN (ngoài session) ghi xong file —
        # phải tự làm mới định kỳ trong lúc đang chạy để "tự hiện kết quả" đúng nghĩa, không
        # bắt người dùng phải tự bấm gì mới thấy cập nhật.
        time.sleep(2)
        st.rerun()
    elif _backtest_failed_current:
        st.error("❌ Lần cập nhật gần nhất bị lỗi. Kết quả đang hiển thị (nếu có) là kết quả cũ.")
        with st.expander("🔍 Chi tiết kỹ thuật", expanded=False):
            st.code(str(_bt_status.get("error")), language="text")
    elif _backtest_stale:
        pass
    elif df_view.empty:
        st.info("ℹ️ Chưa có kết quả đối chiếu nào — hệ thống sẽ tự tính khi có dữ liệu.")

    if not df_view.empty:
        avg_mape = df_view["% Lệch"].mean()
        avg_mae = df_view["Sai lệch"].mean()
        n_samples = len(df_view)
        n_uploads = len(df_view["Upload"].unique()) if "Upload" in df_view.columns else len(file_paths)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            if avg_mape < 7.0:
                status_mape = "MAPE dưới ngưỡng 7%"
                color_mape = "#087762"
            elif avg_mape <= 10.0:
                status_mape = "Cần tiếp tục theo dõi (7-10%)"
                color_mape = "#c77700"
            else:
                status_mape = "Hệ thống đang tự tối ưu mô hình (> 10%)"
                color_mape = "#dc2626"
            st.markdown(f"""
            <div style="border:1px solid #e2e8f0; border-radius:10px; padding:16px; background:#fff;">
                <span style="color:#64748b; font-size:12px; font-weight:600;">MAPE TỔNG THỂ</span>
                <div style="font-size:28px; font-weight:800; margin:4px 0; color:#1e293b;">{avg_mape:.2f}<small style="font-size:14px; font-weight:500; color:#64748b;">%</small></div>
                <span style="color:{color_mape}; font-size:12px; font-weight:700;">● {status_mape}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown(f"""
            <div style="border:1px solid #e2e8f0; border-radius:10px; padding:16px; background:#fff;">
                <span style="color:#64748b; font-size:12px; font-weight:600;">MAE TỔNG THỂ</span>
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
              <span style="color:#087762; font-weight:700;">Xanh (&lt; 7%)</span>: Sai số đang dưới ngưỡng tham khảo ➔ 
              <span style="color:#9b6100; font-weight:700;">Vàng (7–10%)</span>: Cần tiếp tục theo dõi ➔ 
              <span style="color:#dc2626; font-weight:700;">Đỏ (&gt; 10%)</span>: Sai số vượt ngưỡng — Hệ thống tự động tối ưu GUMNet Candidate ngầm.<br>
            • <b>MAE (USD)</b>: Sai số tuyệt đối tính bằng số tiền thực tế (USD/thùng).<br>
            • <b>Bảng nhiệt (Heatmap)</b>: Màu sắc thể hiện mức sai số tương đối giữa các mốc. Các mốc xa có thể có sai số khác mốc gần tùy giai đoạn dữ liệu.
        </div>
        """, unsafe_allow_html=True)

        _col_card_desc, _col_card_act = st.columns([3.2, 1.3])
        with _col_card_desc:
            st.caption("⚙️ **Chủ động tối ưu**: Hệ thống tự động huấn luyện khi MAPE > 10%. Nếu muốn ép máy tối ưu ngay mô hình mới với dữ liệu hiện tại, bạn có thể bấm nút bên cạnh.")
        with _col_card_act:
            if pipeline_engine.already_trained(fingerprint):
                st.caption('Dữ liệu hiện tại đã được huấn luyện và đánh giá. Có thể tối ưu tiếp khi dữ liệu thay đổi.')
            elif st.button("⚡ Tối ưu mô hình ngay", key="btn_force_retrain_page2", disabled=bool(is_pipeline_busy), use_container_width=True):
                fps = [p.name for p in file_paths]
                launch_pipeline_with_feedback("MANUAL-PAGE2", 0, fps, force_retrain=True)
        
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
        • <b>Xem biểu đồ đối chiếu gộp toàn bộ lịch sử:</b> chuyển sang tab <b>"📈 Biểu đồ"</b> ở menu bên trái.
    </div>
    """, unsafe_allow_html=True)

    if _backtest_running_now or _backtest_failed_current or _backtest_stale:
        _col_upd_l2, _col_upd_r2 = st.columns([3, 1])
        with _col_upd_l2:
            if _backtest_running_now:
                st.info("⏳ Đang tự động cập nhật bảng đối chiếu... sẽ tự hiện khi xong, không cần bấm gì.")
                time.sleep(2)
                st.rerun()
            elif _backtest_failed_current:
                st.error("❌ Lần cập nhật gần nhất bị lỗi. Bảng đang hiển thị (nếu có) là kết quả cũ.")
            else:
                st.empty()
        with _col_upd_r2:
            _btn_label2 = "🔁 Thử lại" if _backtest_failed_current else "🔄 Cập nhật (dự phòng)"
            if st.button(_btn_label2, key="btn_update_backtest_p3", use_container_width=True,
                         disabled=_backtest_running_now):
                launch_pipeline_with_feedback(
                    "MANUAL-BACKTEST", 0, [p.name for p in file_paths], force_retrain=False
                )

    st.markdown("#### 📂 Nhật ký các đợt nạp dữ liệu & trạng thái tự động hóa")
    ingest_hist = data_pipeline.get_ingestion_history()
    pipe_status = pipeline_engine.get_pipeline_status()
    
    if ingest_hist:
        hist_table = []
        for h in ingest_hist:
            saved_names = ", ".join([f["filename"] for f in h.get("saved_files", [])]) or "Không có file mới"
            retrain_status = "Đang xử lý ngầm"
            if (pipe_status.get("batch_info") or {}).get("batch_id") == h.get("batch_id"):
                if pipe_status.get("status") == "complete":
                    dec = pipe_status.get("retrain_decision", {})
                    cand = pipe_status.get("candidate_result", {})
                    if cand and cand.get("promoted"):
                        retrain_status = "✓ Đã áp dụng GUMNet candidate mới"
                    elif dec and dec.get("needed"):
                        retrain_status = "Giữ nguyên GUMNet hiện tại"
                    else:
                        retrain_status = "Không cần tối ưu (MAPE tốt)"
                elif pipe_status.get("status") == "failed":
                    retrain_status = "Có lỗi (Giữ GUMNet hiện tại)"
            else:
                retrain_status = "Hoàn tất kiểm định"

            hist_table.append({
                "Mã đợt": h.get("batch_id", "")[:18],
                "Thời gian nạp": h.get("created_at", ""),
                "Tệp đã lưu": saved_names,
                "Số ngày mới": f"{h.get('total_new_rows', 0)} ngày",
                "Kiểm định file": f"✓ {h.get('saved_count', 0)} hợp lệ / {h.get('total_files', 0)} file",
                "Tối ưu GUMNet": retrain_status,
                "Trạng thái": "Đã lưu" if h.get("saved_count", 0) > 0 else "Từ chối"
            })
        df_hist = pd.DataFrame(hist_table)
        safe_dataframe(df_hist)
        
        csv_hist = df_hist.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Xuất Báo Cáo Lịch Sử Nạp Dữ Liệu (CSV)",
            data=csv_hist,
            file_name="Lich_su_cap_nhat_du_lieu.csv",
            mime="text/csv",
            key="btn_dl_hist_full"
        )
    elif file_info:
        history_rows = []
        for idx, fi in enumerate(file_info):
            history_rows.append({
                "Đợt": f"#{idx+1:02d}",
                "Tên tệp tin": fi["name"],
                "Ngày dữ liệu cuối": fi["max_date"].strftime("%d/%m/%Y"),
                "Số dòng dữ liệu": f"{fi['rows']:,} dòng",
                "Trạng thái": "Đã kiểm định"
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
            
        calc_height = min(max(len(sub_up) * 38 + 50, 350), 650)
        safe_dataframe(sub_up[cols_show].style.format({"Dự báo":"{:.2f}","Thực tế":"{:.2f}","Sai lệch":"{:.2f}","% Lệch":"{:.2f}%"}), height=calc_height)
    else:
        st.info("Chưa có dữ liệu lịch sử đối chiếu.")


# ──────────────────────────────────────────
# TRANG 4: BIỂU ĐỒ (gộp toàn bộ lịch sử, tách riêng khỏi trang Lịch sử theo yêu cầu —
# xem đủ 4 mặt hàng cùng lúc, trải dài từ ngày cũ nhất đến mới nhất, không giới hạn ở 1 đợt)
# ──────────────────────────────────────────
elif nav_choice == "📈  Biểu đồ":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="color:#00ad91; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.09em;">TRA CỨU & KIỂM TOÁN</div>
        <h1 style="margin:4px 0; font-size:28px; font-weight:800; letter-spacing:-.03em;">Biểu Đồ Đối Chiếu</h1>
        <p style="margin:0; color:#64748b; font-size:14px;">Gộp toàn bộ các đợt nạp dữ liệu thành 1 chuỗi thời gian liên tục (cũ nhất → mới nhất) cho từng mặt hàng.</p>
    </div>
    <div style="background:#f8fafc; border-left:4px solid #087762; border-radius:6px; padding:11px 16px; margin:14px 0 18px; font-size:13px; color:#334155; line-height:1.6;">
        <b>💡 Cách xem:</b> Chọn 1 chân trời dự báo (horizon) — hệ thống tự vẽ đủ 4 biểu đồ (MG95, MG92, DO 0.001%, DO 0.05%). Đường <b>nét liền xanh ngọc</b> là giá Thực tế, đường <b>nét đứt tím</b> là giá Dự báo. Hai đường càng bám sát nhau chứng tỏ mô hình dự đoán càng chính xác.
    </div>
    """, unsafe_allow_html=True)

    if not df_view.empty:
        sh_full = st.selectbox("Chọn chân trời dự báo:", [f"{h}d" for h in HORIZONS], key="sh_chart_full_page")
        sub_full = df_view[df_view["Horizon"] == sh_full]
        for tgt_full in TARGET_COLS:
            tsub_full = sub_full[sub_full["Target"] == tgt_full]
            if tsub_full.empty:
                continue
            fig_full = go.Figure()
            for m in sel_models:
                ms_full = tsub_full[tsub_full["Model"] == m].sort_values(DATE_COL)
                if not ms_full.empty:
                    fig_full.add_trace(go.Scatter(x=ms_full[DATE_COL], y=ms_full["Dự báo"], name=f"Dự báo ({m})", mode="lines+markers", line=dict(dash="dash", color="#7c3aed")))
            act_full = tsub_full.drop_duplicates(DATE_COL).sort_values(DATE_COL)
            fig_full.add_trace(go.Scatter(x=act_full[DATE_COL], y=act_full["Thực tế"], name="Thực tế", mode="lines+markers", line=dict(color="#00d4aa", width=3)))
            fig_full.update_layout(title=f"{tgt_full} ({sh_full}) — toàn bộ lịch sử", template="plotly_dark", height=380, hovermode="x unified")
            safe_plotly_chart(fig_full)
    else:
        st.info("Chưa có dữ liệu lịch sử đối chiếu.")


# ──────────────────────────────────────────
# TRANG 5: HƯỚNG DẪN SỬ DỤNG
# ──────────────────────────────────────────
elif nav_choice == "❓  Hướng dẫn sử dụng":
    st.markdown("""
    <div style="margin-bottom: 22px;">
        <div style="color:#00ad91; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.09em;">HƯỚNG DẪN VẬN HÀNH</div>
        <h1 style="margin:4px 0; font-size:28px; font-weight:800; letter-spacing:-.03em;">Hướng Dẫn Sử Dụng &amp; Vận Hành Hệ Thống</h1>
        <p style="margin:0; color:#64748b; font-size:14px;">Quy chuẩn dữ liệu đầu vào, cách đọc chỉ số sai số và cơ chế tự động hóa GUMNet ngầm.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:14px 16px; margin-bottom:20px;">
        <div style="font-size:14px; font-weight:700; color:#0f172a; margin-bottom:4px;">Hướng dẫn tương tác</div>
        <div style="font-size:12.5px; color:#64748b; margin-bottom:12px;">Chọn nội dung để hệ thống chuyển đến đúng trang và chỉ dẫn từng thao tác.</div>
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
            <button data-oil-tour="forecast" style="border:1px solid #99f6e4; background:#ecfdf5; color:#087762; border-radius:7px; padding:8px 12px; cursor:pointer; font-weight:600;">Dự báo</button>
            <button data-oil-tour="metrics" style="border:1px solid #bfdbfe; background:#eff6ff; color:#1d4ed8; border-radius:7px; padding:8px 12px; cursor:pointer; font-weight:600;">Đánh giá mô hình</button>
            <button data-oil-tour="history" style="border:1px solid #ddd6fe; background:#f5f3ff; color:#6d28d9; border-radius:7px; padding:8px 12px; cursor:pointer; font-weight:600;">Lịch sử &amp; xuất dữ liệu</button>
            <button data-oil-tour="charts" style="border:1px solid #fed7aa; background:#fff7ed; color:#c2410c; border-radius:7px; padding:8px 12px; cursor:pointer; font-weight:600;">Biểu đồ</button>
            <button id="oil-replay-btn" style="border:1px solid #cbd5e1; background:#ffffff; color:#334155; border-radius:7px; padding:8px 12px; cursor:pointer; font-weight:600;">Xem lại từ đầu</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Khung tải file mẫu và cấu trúc cột bắt buộc
    st.markdown("#### 📥 Tệp Mẫu Dữ Liệu Thị Trường &amp; Cấu Trúc Cột")
    col_t1, col_t2 = st.columns([1.2, 0.8])
    with col_t1:
        st.markdown("""
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:16px; font-size:13px; color:#334155;">
            <b>Bảng tính tải lên cần đáp ứng các điều kiện sau:</b><br>
            • <b>Định dạng hỗ trợ:</b> <code>.xlsx</code>, <code>.xls</code>, <code>.csv</code>.<br>
            • <b>Cột thời gian:</b> Cần có cột <code>Ngày</code> (hoặc <code>Date</code>, <code>ngay</code>).<br>
            • <b>Các cột giá mục tiêu:</b> <code>MG95</code>, <code>MG92</code> (USD/thùng), <code>DO 0.001%</code>, <code>DO 0.05%</code> (USD/tấn). Có thể có 1 hoặc nhiều cột.<br>
            • <b>Kiểu dữ liệu:</b> Nên dùng số thực dương và file không có macro; giá trị âm không được hỗ trợ.<br>
            • <b>Dữ liệu độc lập:</b> Hệ thống kiểm tra từng file riêng biệt. File lỗi sẽ bị từ chối mà không làm ảnh hưởng đến các file hợp lệ khác.
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        template_file = ROOT / "assets" / "file_mau_gia_dau.xlsx"
        if template_file.exists():
            st.markdown("""
            <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:16px; text-align:center;">
                <div style="font-size:32px; margin-bottom:6px;">📊</div>
                <b style="color:#087762; font-size:14px;">Tải Tệp Excel Mẫu Chuẩn</b><br>
                <small style="color:#64748b;">Đã định dạng sẵn cột Ngày và 4 cột giá tiêu chuẩn.</small>
                <div style="height:12px;"></div>
            """, unsafe_allow_html=True)
            st.download_button(
                label="📥 Tải File Mẫu (.xlsx)",
                data=template_file.read_bytes(),
                file_name="file_mau_gia_dau.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_template_btn",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 2. Quy trình vận hành 4 bước khép kín
    st.markdown("""
    #### 🔄 Quy Trình Tự Động Hóa Phía Sau
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:10px 0 24px;">
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:24px; margin-bottom:4px;">📥</div>
            <b style="color:#00ad91; font-size:13px;">1. NẠP DỮ LIỆU</b>
            <p style="font-size:12px; color:#64748b; margin:4px 0 0;">Kéo thả 1 hoặc nhiều file. Hệ thống tự kiểm định schema và phát hiện ngày mới.</p>
        </div>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:24px; margin-bottom:4px;">🔮</div>
            <b style="color:#6954d9; font-size:13px;">2. DỰ BÁO TỨC THÌ</b>
            <p style="font-size:12px; color:#64748b; margin:4px 0 0;">GUMNet Production tạo ngay dự báo 7 mốc (+1d đến +60d) mà không cần đợi huấn luyện.</p>
        </div>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:24px; margin-bottom:4px;">📊</div>
            <b style="color:#087762; font-size:13px;">3. ĐỐI CHIẾU NGẦM</b>
            <p style="font-size:12px; color:#64748b; margin:4px 0 0;">Hệ thống tự động so khớp giá thực tế mới với dự báo cũ để tính MAE/MAPE.</p>
        </div>
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:24px; margin-bottom:4px;">🧠</div>
            <b style="color:#c77700; font-size:13px;">4. TỰ TỐI ƯU GUMNET</b>
            <p style="font-size:12px; color:#64748b; margin:4px 0 0;">Nếu MAPE &gt; 10% và đủ mẫu, candidate tự huấn luyện ngầm và chỉ thay thế khi tốt hơn.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. Hướng dẫn đọc chỉ số & Ý nghĩa màu sắc
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("""
        <div class="guide-card">
            <h3 style="margin:0 0 10px; font-size:16px; color:#0f172a;">📊 Cách đọc chỉ số sai số &amp; Ý nghĩa màu sắc</h3>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0; font-size:13px;">
                <b style="color:#087762;">🟢 Xanh lá (MAPE &lt; 7.0%):</b><br>
                <span style="color:#64748b;">MAPE đang dưới ngưỡng tham khảo 7%. Nên xem thêm số mẫu, MAE và kết quả theo từng mốc trước khi sử dụng.</span>
            </div>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0; font-size:13px;">
                <b style="color:#c77700;">🟡 Màu vàng (7.0% ≤ MAPE ≤ 10.0%):</b><br>
                <span style="color:#64748b;">Cần tiếp tục theo dõi. Sai số vẫn nằm trong ngưỡng chấp nhận được của thị trường.</span>
            </div>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0; font-size:13px;">
                <b style="color:#dc2626;">🔴 Màu đỏ (MAPE &gt; 10.0%):</b><br>
                <span style="color:#64748b;">Hệ thống tự động kích hoạt tiến trình huấn luyện GUMNet Candidate ngầm để thích ứng với biến động mới.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        st.markdown("""
        <div class="guide-card">
            <h3 style="margin:0 0 10px; font-size:16px; color:#0f172a;">🎯 Ý nghĩa các mốc chân trời dự báo (Horizons)</h3>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0; font-size:13px;">
                <b style="color:#00ad91;">+1 ngày &amp; +5 ngày:</b><br>
                <span style="color:#64748b;">Dự báo ngắn hạn phục vụ đặt lệnh mua bán hàng ngày và kế hoạch giao dịch trong tuần.</span>
            </div>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0; font-size:13px;">
                <b style="color:#6954d9;">+10 ngày &amp; +15 ngày &amp; +20 ngày:</b><br>
                <span style="color:#64748b;">Dự báo trung hạn bám theo các kỳ điều hành giá xăng dầu và cân đối tồn kho nửa tháng.</span>
            </div>
            <div style="border-top:1px solid #f1f5f9; padding:8px 0; font-size:13px;">
                <b style="color:#087762;">+30 ngày &amp; +60 ngày:</b><br>
                <span style="color:#64748b;">Dự báo dài hạn phục vụ chiến lược nhập khẩu, hợp đồng tương lai và kế hoạch tài chính quý.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Tài liệu PDF đính kèm
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
    st.markdown("#### ❓ Giải Đáp Thắc Mắc Nghiệp Vụ Thường Gặp (FAQ)")
    
    col_faq1, col_faq2 = st.columns(2)
    with col_faq1:
        st.markdown("""
        <div class="guide-card">
            <b style="color:#0f172a; font-size:14px;">1. Bao lâu nên nạp file dữ liệu mới một lần?</b>
            <p style="color:#64748b; font-size:13px; margin:4px 0 0;">
                Khuyến nghị nạp định kỳ <b>1–2 tuần/lần</b> hoặc sau kỳ điều hành giá xăng dầu để duy trì mốc dữ liệu mới.
            </p>
        </div>
        <div style="height:10px;"></div>
        <div class="guide-card">
            <b style="color:#0f172a; font-size:14px;">2. Tệp tải lên có bắt buộc đủ cả 4 mặt hàng không?</b>
            <p style="color:#64748b; font-size:13px; margin:4px 0 0;">
                Không bắt buộc. Nếu tệp Excel chỉ có giá MG95 và DO 0.05%, hệ thống sẽ xử lý các cột nhận diện được và hiển thị kết quả tương ứng.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_faq2:
        st.markdown("""
        <div class="guide-card">
            <b style="color:#0f172a; font-size:14px;">3. Khi hệ thống đang tự tối ưu mô hình, tôi có xem dự báo được không?</b>
            <p style="color:#64748b; font-size:13px; margin:4px 0 0;">
                Trong lúc tối ưu candidate, màn hình có thể tạm khóa để bảo vệ dữ liệu. Sau khi xử lý hoàn tất, bạn có thể xem và xuất dự báo bằng GUMNet Production hoặc model mới nếu candidate được áp dụng.
            </p>
        </div>
        <div style="height:10px;"></div>
        <div class="guide-card">
            <b style="color:#0f172a; font-size:14px;">4. Đơn vị tiền tệ của các mặt hàng được tính thế nào?</b>
            <p style="color:#64748b; font-size:13px; margin:4px 0 0;">
                Theo cấu hình dữ liệu hiện tại, MG95 và MG92 được hiển thị theo <b>USD/thùng</b>; DO 0.001% và DO 0.05% được hiển thị theo <b>USD/tấn</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
