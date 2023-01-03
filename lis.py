import pandas as pd
import cv2
from IPython.display import Image
from IPython.display import display
import pytesseract
import re
import json
import math
from decimal import Decimal


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


def convert_table_to_dictionary(df):

    # Holds all information about the invoice and is the return value
    invoice_dict = dict()

    df_total = df.loc[df['class'] == 'Invoice-total']
    df_total = df_total.sort_values('conf', ascending=False)

    df_tax = df.loc[df['class'] == 'Tax-rate']
    df_tax = df_tax.sort_values('conf', ascending=False)

    df_line = df.loc[df['class'] == 'Invoice-line']
    df_line = df_line.sort_values('conf', ascending=False)

    if len(df_total.index) < 1:
        print('Validation failed: There must be at leaste one invoice total.')
        return invoice_dict
    elif len(df_line.index) < 1:
        print('Validation failed: There must be at leaste one invoice line.')
        return invoice_dict

    # Convert the classes in the data frame from a json string into a readable format
    # Convert the invoice total to a decimal number with 2 digits
    total_json = df_total['ocr_text'].iat[0]
    total_list = json.loads(total_json)
    total_amount = round(Decimal(total_list[0]), 2)

    # In case a tax rate exits convert it into a decimal number
    tax_rate = 0
    if len(df_tax.index) > 0:
        tax_json = df_tax['ocr_text'].iat[0]
        tax_list = json.loads(tax_json)
        tax_string = tax_list[0]
        # Get tax amount with regex
        pattern = r'\d+(?:\.\d+)?'
        regex_list = re.findall(pattern, tax_string)
        # Check if at least one value was found
        if len(regex_list) < 1:
            print('Validation failed: The tax value could not be determined.')
            return invoice_dict
        else:
            tax_rate = round(Decimal(regex_list[0]) / 100, 2)

    # Convert the invoice lines to decimal number with 2 digits
    lines_list = []
    current_total = 0
    for index, row in df_line.iterrows():
        single_line_dict = dict()
        # Get line amount
        line_list = json.loads(row['ocr_text'])
        line_amount = line_list[len(line_list) - 1]
        regex_list = re.findall(r'\d+(?:\.\d{1,2})?', line_amount)
        if not regex_list:
            continue
        line_amount_decimal = round(Decimal(regex_list[0]), 2)

        # Get description
        line_description = line_list[0]

        # Create line dictionary
        single_line_dict['Description'] = line_description
        single_line_dict['Amount'] = line_amount
        lines_list.append(single_line_dict)

        # Check if all lines have been found yet
        current_total += line_amount_decimal
        current_total_incl_tax = current_total + (current_total * tax_rate)
        current_total_incl_tax = math.floor(current_total_incl_tax * 100) / 100
        # After the current total was added check if it matches the total amount
        # Keep iterating until the current total exactly matches the total amount
        # Due to possibly round erros a difference of 0.01 is exceptable
        if (int(total_amount * 100) >= int(current_total_incl_tax * 100)) and (
                int(total_amount * 100) <= (int(current_total_incl_tax * 100)) + 1):
            invoice_dict['invoice-lines'] = lines_list
            invoice_dict['tax'] = str(tax_rate)
            invoice_dict['total'] = str(total_amount)
            return invoice_dict
        elif int(total_amount * 100) < int(current_total_incl_tax * 100):
            print('Validation failed: Line item amount does not match total amount. (Amount exceeded)')
            return invoice_dict
    # If this line was reached it means there were not enough invoice lines to approximate the total amount
    print('Validation failed: Line item amount does not match total amount. (Amount too low)')
