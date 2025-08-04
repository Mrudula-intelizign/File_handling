import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from io import BytesIO
from typing import Tuple, List
import re
import streamlit as st
from PIL import Image

def clean_table_headers(headers):
    cleaned = []
    seen = set()
    for i, h in enumerate(headers):
        if h is None or h == '':
            h = f"Column_{i}"
        original_h = h
        count = 1
        while h in seen:
            h = f"{original_h}_{count}"
            count += 1
        cleaned.append(h)
        seen.add(h)
    return cleaned


class Trail1:
    def __init__(self, pdf_path_or_bytes):
        if isinstance(pdf_path_or_bytes, bytes):
            self.doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
            self.pdfplumber_doc = pdfplumber.open(BytesIO(pdf_path_or_bytes))
        else:
            self.doc = fitz.open(pdf_path_or_bytes)
            self.pdfplumber_doc = pdfplumber.open(pdf_path_or_bytes)

    def get_page_count(self) -> int:
        return len(self.doc)

    def extract_from_page(self, page_number: int) -> Tuple[str, List[Tuple[int, BytesIO, str]], List[pd.DataFrame], List[str]]:
        if not (0 <= page_number < len(self.doc)):
            raise IndexError("Page number out of range.")

        page = self.doc[page_number]
        text = page.get_text()

        images = []
        page_images = page.get_images(full=True)
        for img_index, img in enumerate(page_images, start=1):
            xref = img[0]
            base_image = self.doc.extract_image(xref)
            image_bytes = base_image['image']
            image_ext = base_image['ext']
            image_io = BytesIO(image_bytes)
            images.append((img_index, image_io, image_ext))

        dataframes = []
        pdfplumber_page = self.pdfplumber_doc.pages[page_number]
        tables = pdfplumber_page.extract_tables()
        for table in tables:
            if table and len(table) > 1:
                raw_headers = table[0]
                clean_headers = clean_table_headers(raw_headers)
                df = pd.DataFrame(table[1:], columns=clean_headers)
                dataframes.append(df)


        table_heading_pattern = re.compile(r'\b(Table|TABLE)[\s\-]*[A-Za-z0-9]+\.?', re.IGNORECASE)
        table_headings = [m.group(0) for m in table_heading_pattern.finditer(text)]

        return text, images, dataframes, table_headings


    def close(self):
        self.doc.close()
        self.pdfplumber_doc.close()


st.set_page_config(layout="wide")
st.title("PDF Page-wise Extractor (Text, Images, Tables)")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()

    extractor = Trail1(pdf_bytes)
    total_pages = extractor.get_page_count()

    st.sidebar.markdown("## Page Navigation")
    page_number = st.sidebar.slider("Select Page", 1, total_pages, 1)
    text, images, tables, table_headings = extractor.extract_from_page(page_number - 1)

    st.subheader(f"Page {page_number} / {total_pages}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Extracted Text")
        st.text_area("Text", text, height=300)

        if table_headings:
            st.markdown("### Table Headings Detected")
            for heading in table_headings:
                st.markdown(f"- {heading}")
        else:
            st.info("No table headings found.")

    with col2:
        st.markdown("### Extracted Images")
        if images:
            for img_index, image_io, ext in images:
                st.image(Image.open(image_io), caption=f"Image {img_index} ({ext})", use_column_width=True)
        else:
            st.info("No images found on this page.")

    st.markdown("### Extracted Tables")
    if tables:
        for idx, df in enumerate(tables, 1):
            st.markdown(f"**Table {idx}:**")
            st.dataframe(df)
    else:
        st.info("No tables found on this page.")

    extractor.close()
else:
    st.info("Upload a PDF to begin.")
