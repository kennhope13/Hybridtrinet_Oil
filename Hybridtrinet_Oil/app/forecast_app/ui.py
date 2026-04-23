# forecast_app/ui.py
import streamlit as st


def ti(name: str) -> str:
    return f'<i class="ti ti-{name}"></i>'


def soft_divider():
    st.markdown('<hr class="hr-soft">', unsafe_allow_html=True)


def page_header():
    st.markdown(
        """
        <div class="hero">
          <div class="page-header">
            <div class="page-title">DỰ ĐOÁN GIÁ XĂNG DẦU</div>
            <div class="page-subtitle">
              Nền tảng dự đoán chuỗi thời gian sử dụng mô hình HybridTriNet (KAN + GRU + Attention)
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(icon_name: str, text: str):
    st.markdown(
        f"""
        <div class="section-h">
          <div class="section-ico">{ti(icon_name)}</div>
          <div class="section-txt">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(title: str, value: str, icon: str = "activity", subtitle: str = ""):
    sub_html = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="row">
            <div class="ttl">{title}</div>
            <div class="badge">{ti(icon)}</div>
          </div>
          <div class="val">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
