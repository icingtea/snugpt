import pdfplumber
from pathlib import Path

CONFIG = [
    {
        "input": Path("data/raw/ug-handbook/ug_handbook.pdf"),
        "output": Path("data/extracted/academics/UG_HANDBOOK.txt"),
    },
    {
        "input": Path("data/raw/students/STUDENT_HANDBOOK.pdf"),
        "output": Path("data/extracted/students/STUDENT_HANDBOOK.txt"),
    },
    {
        "input": Path("data/raw/students/LAUNDRY_SERVICES.pdf"),
        "output": Path("data/extracted/students/LAUNDRY_SERVICES.txt"),
    },
    {
        "input": Path("data/raw/students/OCJ_POLICY.pdf"),
        "output": Path("data/extracted/students/OCJ_POLICY.txt"),
    },
    {
        "input": Path("data/raw/students/SC_CONSTITUTION.pdf"),
        "output": Path("data/extracted/students/SC_CONSTITUTION.txt"),
    },
]


def scrape_pdfs(config_list: list[dict]):
    for item in config_list:
        input_path = item["input"]
        output_path = item["output"]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        text_content = []

        with pdfplumber.open(input_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)

        full_text = "\n".join(text_content)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(full_text)


if __name__ == "__main__":
    scrape_pdfs(CONFIG)
