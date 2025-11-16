import os
import shutil
from pathlib import Path

PREFIX_MAP = {
    "b-tech": "BTech",
    "b-sc": "BSC",
    "b-a": "BA",
    "ba": "BA",
    "bachelor-in-design": "BDES",
    "bachelors-in-design": "BDES",
    "bachelor-of-management-studies": "BMS",
}

def detect_prefix(dirname):
    parts = dirname.split("-")

    first_two = "-".join(parts[:2])
    if first_two in PREFIX_MAP:
        return PREFIX_MAP[first_two], "-".join(parts[2:])

    if parts[0] in PREFIX_MAP:
        return PREFIX_MAP[parts[0]], "-".join(parts[1:])

    return None, dirname

def to_slug_capital_case(slug):
    words = slug.split("-")
    capped = [w.capitalize() for w in words]
    return "-".join(capped)

def detect_file_type(path):
    name = path.stem.lower()
    if "brochure" in name:
        return "brochure"
    if "prospectus" in name:
        return "prospectus"
    return None

def main():
    root = Path("data/raw")

    for program_dir in root.iterdir():
        if not program_dir.is_dir():
            continue

        prefix, program_slug = detect_prefix(program_dir.name)
        if prefix is None:
            continue

        program_name = to_slug_capital_case(program_slug)

        for file in program_dir.iterdir():
            if not file.is_file() or file.suffix.lower() != ".pdf":
                continue

            file_type = detect_file_type(file)
            if file_type is None:
                continue

            new_name = f"{prefix}_{program_name}_{file_type}.pdf"
            dest = root / new_name

            shutil.move(str(file), str(dest))

        try:
            program_dir.rmdir()
        except OSError:
            pass

if __name__ == "__main__":
    main()
