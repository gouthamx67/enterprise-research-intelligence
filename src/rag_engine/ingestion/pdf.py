import pymupdf


def extract_pdf(path):
    pdf = pymupdf.open(path)

    pages = []

    for page_number, page in enumerate(pdf):
        text = page.get_text()

        pages.append({
            "page_number": page_number + 1,
            "text": text
        })

    return pages