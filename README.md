# PDF Page Extractor with Streamlit

This is a Streamlit web application that handles PDF files and displays the following for each page:
-  First 300 words (text summary)
-  Images embedded on the page
-  Tables present in the page

The app processes PDFs page-by-page, helping you get a quick overview of large documents without manually reading everything.

---

## Features

- Extracts and displays:
  - First 300 words from each PDF page
  - All images present on that page
  - Tables detected within the page
- Paginated layout to navigate through pages
- Clean and simple Streamlit UI for ease of use

---

## Tech Stack

- Python 
- Streamlit 
- PyMuPDF or pdfplumber (for PDF parsing)
- PIL / OpenCV (for image extraction)
- Pandas (for table formatting)

## Usage
- Open the app in your browser.
- Upload a PDF file.
- Navigate through pages using the sidebar or pagination controls.
- View text (first 300 words), images, and tables per page.



