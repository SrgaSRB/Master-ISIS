"""Bez API tokena: python genius_scraper.py
Instalacija: python -m pip install requests beautifulsoup4
Preuzima do 20 tekstova sa linkova dostupnih u HTML-u stranice izvođaca.
Ako nedostaju linkovi: python genius_scraper.py --links linkovi.txt
U linkovi.txt stavi po jedan Genius link pesme u svaki red.
"""

import argparse
from datetime import datetime
from pathlib import Path
import re
import time
from urllib.parse import unquote, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup


class AccessBlocked(Exception):
    pass


def fetch(session, url):
    response = session.get(url, timeout=30)
    if response.status_code in (401, 403, 429):
        raise AccessBlocked(f"HTTP {response.status_code}: sajt ogranicava pristup. Prekidam.")
    response.raise_for_status()
    return BeautifulSoup(response.content, "html.parser")


def song_url(value, base):
    parsed = urlsplit(urljoin(base, value.strip()))
    if parsed.scheme == "https" and parsed.hostname == "genius.com" and parsed.path.lower().endswith("-lyrics"):
        return "https://genius.com" + parsed.path
    return None


def extract_lyrics(soup):
    containers = soup.select('[data-lyrics-container="true"]')
    if not containers:
        containers = soup.select(".lyrics")
    parts = []
    for container in containers:
        for unwanted in container.select("script, style, button"):
            unwanted.decompose()
        for br in container.find_all("br"):
            br.replace_with("\n")
        # Ne stavljamo novi red oko linkova: link moze biti usred stiha.
        text = container.get_text(separator="").strip()
        text = re.sub(r"(?m)^\s*\[[^\]\n]+\]\s*$", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Genius HTML scraper bez API tokena")
    parser.add_argument("--artist", default="https://genius.com/artists/Ceca")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--links", type=Path, help="TXT sa linkovima pesama, po jedan u redu")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit mora biti pozitivan")
    artist = urlsplit(args.artist)
    if artist.scheme != "https" or artist.hostname != "genius.com":
        parser.error("--artist mora biti HTTPS Genius URL")

    output = Path(__file__).resolve().parent / ("pesme_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    saved = 0
    with requests.Session() as session:
        session.headers["User-Agent"] = "PersonalLyricsStudy/1.0"
        try:
            if args.links:
                candidates = args.links.read_text(encoding="utf-8-sig").splitlines()
            else:
                page = fetch(session, args.artist)
                candidates = [a["href"] for a in page.select("a[href]")]
            urls = []
            for value in candidates:
                url = song_url(value, args.artist)
                # Ogranicenje na naziv izvođaca u URL-u smanjuje unrelated preporuke.
                if not args.links and url:
                    slug = artist.path.rstrip("/").split("/")[-1].lower()
                    if not urlsplit(url).path.lower().startswith("/" + slug + "-"):
                        continue
                if url and url not in urls:
                    urls.append(url)
            print(f"Pronadjeno linkova: {len(urls)}")
            if not urls:
                print("HTML nema linkove pesama. Dodaj ih u linkovi.txt i koristi --links linkovi.txt.")
                return

            for url in urls:
                if saved >= args.limit:
                    break
                try:
                    song_page = fetch(session, url)
                    lyrics = extract_lyrics(song_page)
                    if not lyrics:
                        print(f"Preskacem bez teksta: {url}")
                        continue

                    output.mkdir(parents=True, exist_ok=True)
                    slug = unquote(urlsplit(url).path.rstrip("/").split("/")[-1])
                    filename = re.sub(r'[<>:"/\\|?*]+', "_", slug).strip(" ._")
                    if not filename:
                        filename = f"pesma_{saved + 1}"
                    destination = output / f"{filename}.txt"
                    destination.write_text(lyrics + "\n", encoding="utf-8")
                    saved += 1
                    print(f"[{saved}] Sacuvano: {destination}")
                    time.sleep(1)
                except AccessBlocked as e:
                    print(e)
                    break
                except requests.RequestException as e:
                    print(f"Greska pri preuzimanju {url}: {e}")
                except OSError as e:
                    print(f"Greska pri cuvanju {url}: {e}")

        except AccessBlocked as e:
            print(e)
        except (OSError, requests.RequestException) as e:
            print(f"Greska: {e}")

    if saved:
        print(f"Ukupno sacuvano: {saved}. Folder: {output}")
    else:
        print("Nijedna pesma nije sacuvana.")


if __name__ == "__main__":
    main()
        