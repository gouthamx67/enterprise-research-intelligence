from src.rag_engine.core.document import Block
from src.rag_engine.preprocessing.sections import build_sections


def create_blocks():

    return [
        Block(
            block_number=0,
            text="Introduction",
            bbox=(0, 50, 300, 80),
            page_number=1,
            is_heading=True,
        ),
        Block(
            block_number=1,
            text="The company develops enterprise software.",
            bbox=(0, 100, 500, 150),
            page_number=1,
        ),
        Block(
            block_number=2,
            text="The company serves customers globally.",
            bbox=(0, 160, 500, 210),
            page_number=1,
        ),
        Block(
            block_number=3,
            text="Financial Results",
            bbox=(0, 250, 300, 280),
            page_number=2,
            is_heading=True,
        ),
        Block(
            block_number=4,
            text="Revenue increased by 18%.",
            bbox=(0, 300, 500, 350),
            page_number=2,
        ),
        Block(
            block_number=5,
            text="Operating income increased by 12%.",
            bbox=(0, 360, 500, 410),
            page_number=2,
        ),
    ]


def main():

    blocks = create_blocks()

    sections = build_sections(blocks)

    print("SECTIONS")
    print("========")

    for section in sections:

        print()
        print("Title:", section.title)
        print("Level:", section.level)
        print("Start page:", section.start_page)
        print("End page:", section.end_page)

        print("Blocks:")

        for block in section.blocks:
            print("  -", block.text)


if __name__ == "__main__":
    main()