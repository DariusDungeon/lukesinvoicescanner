import pandas as pd
import cv2
from IPython.display import Image
from IPython.display import display
import pytesseract
import re
import json


def convert_to_table(invoice, processing_folder, debug):
    # Read data
    # invoice = 'INV00000555'
    classes = {
        0: "Invoice-line",
        1: "Invoice-total",
        2: "Tax-rate"}
    config = {
        "Invoice-line": '-c preserve_interword_spaces=1 --psm 6',
        "Invoice-total": '-c preserve_interword_spaces=0 --psm 6',
        "Tax-rate": '-c preserve_interword_spaces=0 --psm 6'}

    headers = ['class', 'x', 'y', 'width', 'height', 'conf']
    df = pd.read_csv(f'{processing_folder}/labels/{invoice}.txt', delimiter=' ', header=None, names=headers)
    df['class'] = df['class'].map(classes)
    df = df.assign(ocr_text=None)

    cv2.IMREAD_GRAYSCALE = 1
    for cls in classes.values():
        cnt = 0
        base_path = f'/content/lukesinvoicescanner/processing/exp/crops/{cls}/{invoice}'
        for index, row in df.iterrows():
            # Continue with OCR if current class from list equals the class name from the data frame
            if row['class'] != cls:
                continue
            cnt += 1
            img_path = ''
            # Determine the correct path name format of the image crop provided by yolov5
            if cnt > 1:
                img_path = f'{base_path}{cnt}.jpg'
            else:
                img_path = f'{base_path}.jpg'

            # Use tesseract to obtain the text from the image crop
            img = cv2.imread(img_path)
            ocr_text = pytesseract.image_to_string(img, lang='eng', config=config[cls]).replace("\n", "").replace(
                '\x0c', '')

            # Display ocr text and image if enabled
            if debug:
                print(ocr_text)
                display(Image(filename=img_path, height=23))
            # Split in case of multiple white spaces
            pattern = r'\s\s+'
            ocr_list = re.split(pattern, ocr_text)
            df.at[index, 'ocr_text'] = json.dumps(ocr_list)
    return df
