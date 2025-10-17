import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from io import BytesIO
from typing import Tuple, List
import re
from PIL import Image


def clean_table_headers(headers): #returns a cleaned list of column names
    cleaned = [] # Initializes an empty list where cleaned header names will be appended
    seen = set() # Keeps track of header names already used so we can avoid duplicates.
    for i, h in enumerate(headers): # Loop through headers; i is index (0,1,2...) and h is the header value.
        if h is None or h == '': # If header cell is None or empty string — i.e., missing header — then:
            h = f"Column_{i}" # Replace missing header with a fallback name "Column_0", "Column_1", etc.
        original_h = h # Save the original header
        count = 1 # Start a counter to disambiguate duplicate names.
        while h in seen: # If current header h was already used (duplicate), then create a unique variant:
            h = f"{original_h}_{count}" # Append _1, _2, etc. to make a unique header name.
            count += 1 # Increment the suffix counter.
        cleaned.append(h) # Add the final unique header to the cleaned list.
        seen.add(h) # Record that this header name is now used.
    return cleaned # Return thet this header name is now used.


class Trail9:
    """
    PDF Extractor class using PyMuPDF (fitz) and pdfplumber.

    Capabilities:
      ✅ Extract full text
      ✅ Extract images
      ✅ Extract tables as pandas DataFrames
      ✅ Detect table headings like 'Table 1', 'TABLE A', etc.
    """

    def __init__(self, pdf_path_or_bytes): # Constructor. Accepts either bytes (raw PDF content) or a file path (string).
        if isinstance(pdf_path_or_bytes, bytes): # Checks if the input is raw bytes (e.g., from an uploaded file). If yes:
            self.doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf") # Opens the PDF in PyMuPDF from the bytes stream;
            self.pdfplumber_doc = pdfplumber.open(BytesIO(pdf_path_or_bytes)) # Opens the same PDF with pdfplumber by wrapping bytes in BytesIO
        else: # If input is not bytes, assume it’s a path/filename:
            self.doc = fitz.open(pdf_path_or_bytes) # Open via path with PyMuPDF.
            self.pdfplumber_doc = pdfplumber.open(pdf_path_or_bytes) # Open via path with pdfplumber.

    def get_page_count(self) -> int:
        """Returns total number of pages in the PDF."""
        return len(self.doc)

    def extract_from_page(
        self, page_number: int
    ) -> Tuple[str, List[Tuple[int, BytesIO, str]], List[pd.DataFrame], List[str]]:
        """
        Extracts text, images, tables, and table headings from a given page.

        Returns:
            - text (str): Extracted text from the page.
            - images (List[Tuple[int, BytesIO, str]]): List of (index, BytesIO, ext).
            - dataframes (List[pd.DataFrame]): Extracted tables.
            - table_headings (List[str]): Detected table heading strings.
        """
        if not (0 <= page_number < len(self.doc)): # Guard to ensure page_number is within range (0..pages-1).
            raise IndexError("Page number out of range.") # Raises an IndexError if the requested page doesn't exist.

        page = self.doc[page_number] # Get the fitz.Page object for the requested page.
        text = page.get_text() # Extract page text using PyMuPDF's .get_text()

        # --- Extract Images --- 
        images = [] # Initialize a list that will hold extracted image info.
        page_images = page.get_images(full=True) # Returns a list of images referenced on the page.
        for img_index, img in enumerate(page_images, start=1): # Iterate over images.
            xref = img[0] # Get the XREF (image object reference) for extraction.
            base_image = self.doc.extract_image(xref) # Returns a dict like {'image': b'...', 'ext': 'png', ...}.
            image_bytes = base_image["image"] # Binary img data
            image_ext = base_image["ext"] # File extension (e.g., 'png', 'jpeg') — used later for captions/format.
            image_io = BytesIO(image_bytes) # Wrap raw bytes in a BytesIO object so Streamlit can read it like a file.
            images.append((img_index, image_io, image_ext)) # Append a tuple (index, BytesIO, extension) for later use (displaying).

        # --- Extract Tables ---
        dataframes = [] # Prepare list to store pandas DataFrames for each detected table.
        pdfplumber_page = self.pdfplumber_doc.pages[page_number] # Get the page object from pdfplumber; it provides extract_tables().
        tables = pdfplumber_page.extract_tables() # pdfplumber attempts to detect tables and returns a list of tables.
        for table in tables: # iterate each detected table
            if table and len(table) > 1: #Only process tables with at least two rows
                raw_headers = table[0] # Treat first row as header row.
                clean_headers = clean_table_headers(raw_headers) # Clean and deduplicate header names using the helper function.
                df = pd.DataFrame(table[1:], columns=clean_headers) #Convert remaining rows into a pandas DataFrame with the cleaned headers.
                dataframes.append(df) #Append DataFrame to list.

        # --- Detect Table Headings ---
        table_heading_pattern = re.compile(
            r"\b(Table|TABLE)[\s\-]*[A-Za-z0-9]+\.?", re.IGNORECASE
        )
        table_headings = [m.group(0) for m in table_heading_pattern.finditer(text)]
        # Find all non-overlapping matches in the extracted text and collect the matched strings (the headings).
        return text, images, dataframes, table_headings

    def extract_full_document(
        self,
    ) -> Tuple[str, List[Tuple[int, BytesIO, str]], List[pd.DataFrame], List[str]]:
        """
        Extracts text, images, tables, and headings from the entire PDF.
        Useful when you want to process the full document in one go.
        """
        full_text = ""
        all_images = []
        all_tables = []
        all_headings = []

        for page_num in range(len(self.doc)):
            text, images, tables, headings = self.extract_from_page(page_num)
            full_text += f"\n--- Page {page_num+1} ---\n" + text
            all_images.extend(images)
            all_tables.extend(tables)
            all_headings.extend(headings)

        return full_text, all_images, all_tables, all_headings

    def close(self):
        """Close all open document handlers."""
        self.doc.close()
        self.pdfplumber_doc.close()
