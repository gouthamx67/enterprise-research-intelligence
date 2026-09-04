from src.rag_engine.core.document import Block, Page

from src.rag_engine.preprocessing.cleaning import (
    find_repeated_blocks,
    classify_repeated_blocks,
    remove_classified_headers_footers,
)


def create_test_pages():

    pages = [
        Page(
            page_number=1,
            blocks=[
                Block(
                    block_number=0,
                    text="ACME CORPORATION",
                    bbox=(0, 40, 300, 60),
                    page_number=1,
                ),
                Block(
                    block_number=1,
                    text="Introduction",
                    bbox=(0, 200, 300, 230),
                    page_number=1,
                ),
                Block(
                    block_number=2,
                    text="Page Footer",
                    bbox=(0, 750, 300, 770),
                    page_number=1,
                ),
            ],
        ),
        Page(
            page_number=2,
            blocks=[
                Block(
                    block_number=0,
                    text="ACME CORPORATION",
                    bbox=(0, 40, 300, 60),
                    page_number=2,
                ),
                Block(
                    block_number=1,
                    text="Financial Results",
                    bbox=(0, 200, 300, 230),
                    page_number=2,
                ),
                Block(
                    block_number=2,
                    text="Page Footer",
                    bbox=(0, 750, 300, 770),
                    page_number=2,
                ),
            ],
        ),
        Page(
            page_number=3,
            blocks=[
                Block(
                    block_number=0,
                    text="ACME CORPORATION",
                    bbox=(0, 40, 300, 60),
                    page_number=3,
                ),
                Block(
                    block_number=1,
                    text="Risk Factors",
                    bbox=(0, 200, 300, 230),
                    page_number=3,
                ),
                Block(
                    block_number=2,
                    text="Page Footer",
                    bbox=(0, 750, 300, 770),
                    page_number=3,
                ),
            ],
        ),
    ]

    page_heights = [800, 800, 800]

    return pages, page_heights


def main():

    pages, page_heights = create_test_pages()

    print("BEFORE")
    print("======")

    for page in pages:

        print(f"Page {page.page_number}")

        for block in page.blocks:
            print(f"  - {block.text}")

    classification = classify_repeated_blocks(
        pages,
        page_heights,
    )

    print()
    print("CLASSIFICATION")
    print("==============")

    print("Headers:")

    for text in sorted(classification["headers"]):
        print(f"  - {text}")

    print("Footers:")

    for text in sorted(classification["footers"]):
        print(f"  - {text}")

    print("Other repeated:")

    for text in sorted(classification["other_repeated"]):
        print(f"  - {text}")

    cleaned_pages = remove_classified_headers_footers(
        pages,
        classification,
    )

    print()
    print("AFTER")
    print("=====")

    for page in cleaned_pages:

        print(f"Page {page.page_number}")

        for block in page.blocks:
            print(f"  - {block.text}")


if __name__ == "__main__":
    main()