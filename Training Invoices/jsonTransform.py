import json
import os

os.chdir(r'/Users/compudime/Documents/TrainingInvoices')

# TrainingInvoices is a folder containing JSON files
invoices_folder = os.getcwd()
for filename in os.listdir():
    if filename.endswith(".json"):
        filepath = os.path.join(invoices_folder, filename)
        with open(filepath, 'r') as f:
            invoice = f.read()
        if isinstance(invoice, str):
            data = json.loads(invoice)
        else:
            data = invoice

        if 'Ch' not in data:
            print(f"Skipping {filename}: no 'Ch' key")
            continue
        rows = []

        for item in data['Ch']:
            if 'Ch' in item:
                for row in item['Ch']:
                    json_row={}
                    for item in row['Cells']:
                        if 'Val' in item:
                            json_row[item['Name']]=item['Val']
                    json_row['SKU'] = json_row.pop('UPC', None)
                    json_row.pop('calcValue', None)
                    json_row.pop('calcValue (2)', None)
                    json_row.pop('Invalid Line', None)
                    json_row.pop('Error Fields', None)
                    rows.append(json_row)
        
        json_data = {}

        for item in data['Ch']:
            if 'Val' in item:
                json_data[item['Name']] = item['Val']

        json_data['Rows'] = rows
        json_data.pop('Batch ID', None)
        json_data.pop('Global Vendor Code', None)
        json_data.pop('Link', None)
        json_data.pop('COG', None)
        json_data.pop('Invalid Invoice', None)
        json_data.pop('Error Message', None)
        json_data.pop('Error Fields', None)
        json_data.pop('Type', None)
        json_data.pop('File Name', None)
        json_data['Adjustment'] = json_data.pop('Surcharge', None)

        output_filename = os.path.splitext(filename)[0] + "_transformed.json"
        output_filepath = os.path.join(invoices_folder, output_filename)
        with open(output_filepath, 'w') as f:
            json.dump(json_data, f, indent=4)  
