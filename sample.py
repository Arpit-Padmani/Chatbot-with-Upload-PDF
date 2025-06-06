import pdfplumber
from PyPDF2 import PdfReader
from pathlib import Path

# Open the PDF file
# with pdfplumber.open("ilovepdf_merged.pdf") as pdf:
#     for page_number, page in enumerate(pdf.pages, start=1):
#         tables = page.extract_tables()
        # print(tables)

        # for table_index, table in enumerate(tables):
        #     print(f"\nPage {page_number} - Table {table_index + 1}")
        #     for row in table:
        #         print(row)


def get_pdf_text_and_tables(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        print(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
            # Reset file pointer for pdfplumber (if needed)
        pdf.seek(0)
        with pdfplumber.open(pdf) as plumber_pdf:
            for page in plumber_pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        row_text = ' | '.join(cell.strip() if cell else '' for cell in row)
                        text += row_text + "\n"
    return text

with open("ilovepdf_merged.pdf", "rb") as f:
    full_text = get_pdf_text_and_tables([f])
    print(full_text)