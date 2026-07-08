import pathlib
import re


def main() -> None:
    js_path = pathlib.Path("index-ASoSUvJW.js")
    content = js_path.read_text(encoding="utf-8", errors="ignore")
    urls = sorted(set(re.findall(r"https?://[^\s\"'`]+", content)))
    for url in urls:
        print(url)


if __name__ == "__main__":
    main()
