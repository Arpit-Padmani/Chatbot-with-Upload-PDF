import pdfplumber
from PyPDF2 import PdfReader
from pathlib import Path
import pandas as pd

def get_pdf_text(pdf_docs):
    text = ""
    tables_dataframes = []

    for pdf in pdf_docs:
        # Extract text from PDF (PyPDF2)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Reset the file pointer for pdfplumber
        pdf.seek(0)

        with pdfplumber.open(pdf) as plumber_pdf:
            for page in plumber_pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    # Append table as a DataFrame
                    df = pd.DataFrame(table)
                    tables_dataframes.append(df)

                    # Optional: also append text version of the table
                    for row in table:
                        row_text = ' | '.join(cell.strip() if cell else '' for cell in row)
                        text += row_text + "\n"

    return text, tables_dataframes