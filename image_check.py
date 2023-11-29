import easyocr 
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2 
import random 
import matplotlib.pyplot as plt

reader = easyocr.Reader(['ko', 'en'], gpu=False)
result = reader.readtext("Training/01.원천데이터/TS_13.제주/JJ_FF01_M0004_1953819_6.jpg")
img    = cv2.imread("Training/01.원천데이터/TS_13.제주/JJ_FF01_M0004_1953819_6.jpg")
img = Image.fromarray(img)
font = ImageFont.truetype("/Applications/XAMPP/xamppfiles/htdocs/php_pro/php.pj/font/SKYBORI.ttf",90)
draw = ImageDraw.Draw(img)
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(256, 3),dtype="uint8")
for i in result :
    x = i[0][0][0] 
    y = i[0][0][1] 
    w = i[0][1][0] - i[0][0][0] 
    h = i[0][2][1] - i[0][1][1]

    color_idx = random.randint(0,255) 
    color = [int(c) for c in COLORS[color_idx]]
    draw.rectangle(((x, y), (x+w, y+h)), outline=tuple(color), width=2)
    draw.text((int((x + x + w) / 2) , y-2),str(i[1]), font=font, fill=tuple(color),)
plt.imshow(img)
plt.show()