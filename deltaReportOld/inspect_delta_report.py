from playwright.sync_api import sync_playwright


URL = "https://poly-pnl.it9.win/delta-report-v3"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            http_credentials={"username": "mm", "password": "2047"},
            accept_downloads=True,
        )
        page = context.new_page()
        page.goto(URL, wait_until="networkidle", timeout=120000)
        page.wait_for_timeout(3000)

        # Snapshot useful metadata for robust selector authoring.
        page.screenshot(path="delta_report_page.png", full_page=True)
        title = page.title()
        print(f"TITLE: {title}")
        print("URL:", page.url)

        print("\nBUTTONS:")
        for b in page.locator("button").all()[:120]:
            text = b.inner_text().strip()
            if text:
                print("-", text)

        print("\nINPUTS:")
        for i in page.locator("input").all()[:200]:
            print(
                "-",
                {
                    "type": i.get_attribute("type"),
                    "name": i.get_attribute("name"),
                    "id": i.get_attribute("id"),
                    "placeholder": i.get_attribute("placeholder"),
                    "value": i.input_value() if (i.get_attribute("type") or "") != "checkbox" else None,
                },
            )

        print("\nSELECTS:")
        for s in page.locator("select").all()[:60]:
            print("-", {"name": s.get_attribute("name"), "id": s.get_attribute("id")})

        print("\nLABELS (first 120):")
        labels = page.locator("label").all()
        for lb in labels[:120]:
            txt = lb.inner_text().strip()
            if txt:
                print("-", txt)

        browser.close()


if __name__ == "__main__":
    main()
