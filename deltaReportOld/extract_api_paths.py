import pathlib
import re


def main() -> None:
    js_path = pathlib.Path("index-ASoSUvJW.js")
    content = js_path.read_text(encoding="utf-8", errors="ignore")
    quoted_strings = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', content)
    quoted_strings += re.findall(r"'([^'\\]*(?:\\.[^'\\]*)*)'", content)

    needles = ("api", "report", "delta", "group", "filter", "date", "pnl", "settle")
    matches = sorted(
        {
            s
            for s in quoted_strings
            if len(s) < 300 and any(n in s.lower() for n in needles)
        }
    )

    out = pathlib.Path("bundle_string_matches.txt")
    out.write_text("\n".join(matches), encoding="utf-8")
    print(f"Wrote {len(matches)} strings to {out}")


if __name__ == "__main__":
    main()
