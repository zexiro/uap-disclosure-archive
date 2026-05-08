.PHONY: all parse download ocr links vault index serve clean refetch-csv

all: parse download ocr links vault index
	@echo "✓ Pipeline complete. Run 'make serve' to browse."

links:
	@python3 scripts/build_links.py

refetch-csv:
	@./scripts/fetch.sh "https://www.war.gov/Portals/1/Interactive/2026/UFO/uap-csv.csv" raw/csv/uap-csv.csv
	@./scripts/fetch.sh "https://www.war.gov/UFO/" raw/ufo-page.html

parse:
	@python3 scripts/parse_csv.py

download:
	@python3 scripts/download.py

ocr:
	@python3 scripts/ocr.py

vault:
	@python3 scripts/build_vault.py

index:
	@python3 scripts/build_search_index.py

serve:
	@echo "→ http://localhost:8765/ui/"
	@cd $(CURDIR) && python3 -m http.server 8765

clean:
	rm -rf raw/text raw/docs_ocr vault ui/search-index.json
