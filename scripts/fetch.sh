#!/usr/bin/env bash
# Wrapper around curl that uses browser-like headers so Akamai doesn't 403 us.
# Usage: fetch.sh <url> <output-path>
set -euo pipefail

UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"

url="$1"
out="$2"
mkdir -p "$(dirname "$out")"

curl -fsSL --compressed \
  -A "$UA" \
  -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8" \
  -H "Accept-Language: en-US,en;q=0.9" \
  -H "Sec-Fetch-Dest: document" \
  -H "Sec-Fetch-Mode: navigate" \
  -H "Sec-Fetch-Site: none" \
  -H "Referer: https://www.war.gov/UFO/" \
  -o "$out" \
  "$url"
