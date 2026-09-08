# -*- coding: utf-8 -*-
"""
Sinh file favicon.png theo Mẫu 1: Thùng phuy kim loại 3D công nghiệp chuẩn ISO.
Có 2 gờ dập nổi (rolling hoops), nắp viền kim loại và van ren trên đỉnh.
"""
from PIL import Image, ImageDraw
import math

SIZE = 256
img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))

def S(val):
    return int(val * 2.56)

x_l, x_r = S(20), S(80)
y_t, y_b = S(28), S(74)

# 1. Vẽ thân thùng phuy hình trụ với gradient kim loại cong
for y in range(y_t, y_b + 1):
    ratio_y = (y - y_t) / (y_b - y_t)
    for x in range(x_l, x_r + 1):
        ratio_x = (x - x_l) / (x_r - x_l)
        # Highlight ánh sáng ở 35% từ trái sang
        light = math.cos((ratio_x - 0.35) * math.pi)
        light = max(0.0, light)
        
        base_r = int(0 + 40 * light * (1 - ratio_y * 0.4))
        base_g = int(100 + 73 * light * (1 - ratio_y * 0.4))
        base_b = int(80 + 65 * light * (1 - ratio_y * 0.4))
        img.putpixel((x, y), (base_r, base_g, base_b, 255))

draw = ImageDraw.Draw(img)

# 2. Đáy thùng phuy (vòm cong dưới)
draw.ellipse([x_l, y_b - S(7), x_r, y_b + S(7)], fill=(5, 71, 61, 255), outline=(0, 100, 80, 255), width=S(1.5))

# 3. Hai gờ dập nổi gia cường kim loại đặc trưng (Rolling Hoops)
g1_y = S(44)
draw.ellipse([x_l - S(1.5), g1_y - S(5), x_r + S(1.5), g1_y + S(5)], outline=(45, 212, 191, 255), width=S(3))
draw.arc([x_l - S(1.5), g1_y - S(5), x_r + S(1.5), g1_y + S(5)], start=0, end=180, fill=(0, 60, 50, 255), width=S(2))

g2_y = S(59)
draw.ellipse([x_l - S(1.5), g2_y - S(5), x_r + S(1.5), g2_y + S(5)], outline=(45, 212, 191, 255), width=S(3))
draw.arc([x_l - S(1.5), g2_y - S(5), x_r + S(1.5), g2_y + S(5)], start=0, end=180, fill=(0, 60, 50, 255), width=S(2))

# 4. Nắp thùng phuy (vành kim loại trên cùng)
draw.ellipse([x_l, y_t - S(8), x_r, y_t + S(8)], fill=(20, 184, 166, 255), outline=(15, 118, 110, 255), width=S(2))
# Lòng nắp lõm xuống
draw.ellipse([x_l + S(5), y_t - S(5.5), x_r - S(5), y_t + S(5.5)], fill=(0, 93, 79, 255))

# 5. Nắp van ren kim loại (Bung cap) trên đỉnh
van_x, van_y = S(62), S(26)
draw.ellipse([van_x - S(4), van_y - S(3), van_x + S(4), van_y + S(3)], fill=(226, 232, 240, 255), outline=(15, 118, 110, 255), width=S(1))
draw.ellipse([van_x - S(1.5), van_y - S(1), van_x + S(1.5), van_y + S(1)], fill=(51, 65, 85, 255))

# Van nhỏ thông hơi bên trái
van2_x, van2_y = S(38), S(28)
draw.ellipse([van2_x - S(2.5), van2_y - S(2), van2_x + S(2.5), van2_y + S(2)], fill=(148, 163, 184, 255), outline=(15, 118, 110, 255), width=S(1))

img.save('D:/Anh_Thuy/assets/favicon.png')
print("Successfully generated 3D Oil Drum Favicon (Mau 1)")
