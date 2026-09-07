import markdown
import subprocess
import os
from pathlib import Path

md_path = Path(r"D:\Anh_Thuy\HUONG_DAN_CAU_HINH_VA_TRIEN_KHAI.md")
html_path = Path(r"D:\Anh_Thuy\temp_doc.html")
pdf_path = Path(r"D:\Anh_Thuy\HUONG_DAN_TRIEN_KHAI_VA_SU_DUNG.pdf")

with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])

template = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Bản Đề Xuất & Yêu Cầu Kỹ Thuật Triển Khai</title>
<style>
    @page {
        size: A4;
        margin: 15mm 15mm 15mm 15mm;
    }
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.5;
        color: #24292e;
        background: #ffffff;
        font-size: 13px;
    }
    h1 {
        color: #1e293b;
        font-size: 17px;
        border-bottom: 2px solid #4f46e5;
        padding-bottom: 6px;
        margin-top: 0;
        text-transform: uppercase;
    }
    h2 {
        color: #334155;
        font-size: 14.5px;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px;
        margin-top: 16px;
        margin-bottom: 8px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: 12px;
    }
    th, td {
        border: 1px solid #cbd5e1;
        padding: 6px 8px;
        text-align: left;
    }
    th {
        background-color: #f1f5f9;
        color: #1e293b;
        font-weight: 600;
    }
    blockquote {
        border-left: 4px solid #4f46e5;
        background-color: #f8fafc;
        margin: 10px 0;
        padding: 6px 10px;
        color: #334155;
        font-size: 12px;
    }
    ul, ol {
        padding-left: 18px;
        margin: 4px 0;
    }
    li {
        margin-bottom: 3px;
    }
    strong {
        color: #0f172a;
    }
    img {
        max-width: 95%;
        max-height: 340px;
        height: auto;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        display: block;
        margin: 8px auto 4px auto;
        page-break-inside: avoid;
    }
    em {
        color: #64748b;
        font-size: 11px;
        display: block;
        text-align: center;
        margin-bottom: 12px;
    }
</style>
</head>
<body>
__BODY__
</body>
</html>"""

full_html = template.replace("__BODY__", html_body)

with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(chrome_path):
    chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

cmd = [
    chrome_path,
    "--headless=new",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={str(pdf_path)}",
    str(html_path)
]
subprocess.run(cmd, check=True)

if html_path.exists():
    html_path.unlink()

print(f"PDF updated successfully at: {pdf_path}")
