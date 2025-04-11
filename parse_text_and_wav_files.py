import os
import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# Precompile regex patterns for efficiency
START_PATTERN = re.compile(r'\\start (.*?)\n')
TX_PATTERN = re.compile(r'\\tx (.*?)\n')
FT_PATTERN = re.compile(r'\\ft(?:i)? (.*)')

cur_dir = os.getcwd()

def fetch_url(url):
    """Fetch the content of a URL and return the BeautifulSoup object."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def extract_href_list(soup):
    """Extract href links from the specified div class."""
    if not soup:
        return []
    field = soup.find_all("div", class_="views-field views-field-fgs-label-s")
    return [f"https://archive.mpi.nl{div.find('a').get('href')}" for div in field if div.find("a")]

def extract_tbt_and_wav_links(href_list):
    """Extract TBT links and corresponding WAV links."""
    def process_href(href):
        soup = fetch_url(href)
        if not soup:
            return None, None
        tbt_link, wav_link = None, None

        # Find TBT link
        links = soup.find_all("div", class_="flat-compound-child")
        if len(links) >= 2:
            tbt = links[1].find('a')
            if tbt:
                tbt_link = f"https://archive.mpi.nl{tbt.get('href')}"

        # Find WAV link
        download_links = soup.find_all("div", class_="flat-compound-download")
        if len(download_links) >= 3:
            wav = download_links[2].find('a')
            if wav:
                wav_link = f"https://archive.mpi.nl{wav.get('href')}"

        return tbt_link, wav_link

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_href, href_list))

    tbt_links, wav_links = zip(*[result for result in results if result != (None, None)])
    return list(tbt_links), list(wav_links)

# Counter global
global_counter = 0

def download_wav(url, global_index):
    """Download a WAV file and return its filename."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Define filename using global index
        filename = f"audio/audio_{global_index + 1}.wav"
        # os.makedirs(filename, exist_ok=True)

        # Save file
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {filename}")
        return filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def parse_segments(tbt_links, wav_filenames):
    """Parse segments and include WAV filenames."""
    mkn_segment_sentence, eng_segment_sentence, start_segment_sentence, wav_segment_files = [], [], [], []

    def process_tbt_link(tbt_link, wav_filename):
        soup = fetch_url(tbt_link)
        if not soup:
            return [], [], [], []

        plain_text_div = soup.find("div", class_="plain-text")
        if not plain_text_div:
            return [], [], [], []

        text_segments = plain_text_div.find_all("p")
        temp_mkn, temp_eng, temp_start, temp_wav = [], [], [], []

        for segment in text_segments:
            start_match = START_PATTERN.search(segment.text)
            tx_match = TX_PATTERN.search(segment.text)
            ft_match = FT_PATTERN.search(segment.text)

            start_text = start_match.group(1).strip() if start_match else ""
            tx_text = tx_match.group(1).strip() if tx_match else ""
            ft_text = ft_match.group(1).strip() if ft_match else ""

            start_text = re.sub(r'\s+', ' ', start_text)
            tx_text = re.sub(r'\s+', ' ', tx_text)
            ft_text = re.sub(r'\s+', ' ', ft_text)

            temp_mkn.append(tx_text)
            temp_eng.append(ft_text)
            temp_start.append(start_text)
            temp_wav.append(wav_filename)

        return temp_mkn, temp_eng, temp_start, temp_wav

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_tbt_link, tbt_links, wav_filenames))

    for temp_mkn, temp_eng, temp_start, temp_wav in results:
        mkn_segment_sentence.extend(temp_mkn)
        eng_segment_sentence.extend(temp_eng)
        start_segment_sentence.extend(temp_start)
        wav_segment_files.extend(temp_wav)

    return mkn_segment_sentence, eng_segment_sentence, start_segment_sentence, wav_segment_files

def main(base_url):
    global global_counter  # Referensi counter global

    mkn_segment_sentence_full, eng_segment_sentence_full, start_segment_sentence_full, wav_segment_files_full = [], [], [], []

    for i in range(5):
        url = base_url if i == 0 else f"{base_url}?page={i}"
        print(f"=== Parsing page {i} ===")

        soup = fetch_url(url)
        href_list = extract_href_list(soup)
        tbt_links, wav_links = extract_tbt_and_wav_links(href_list)

        # Download WAV files
        with ThreadPoolExecutor() as executor:
            wav_filenames = list(
                executor.map(download_wav, wav_links, range(global_counter, global_counter + len(wav_links)))
            )

        # Update counter globally
        global_counter += len(wav_links)

        mkn_segment_sentence, eng_segment_sentence, start_segment_sentence, wav_segment_files = parse_segments(tbt_links, wav_filenames)

        mkn_segment_sentence_full.extend(mkn_segment_sentence)
        eng_segment_sentence_full.extend(eng_segment_sentence)
        start_segment_sentence_full.extend(start_segment_sentence)
        wav_segment_files_full.extend(wav_segment_files)

        print(f"=== Finished parsing page {i} ===")

    df = pd.DataFrame({
        'mkn_segment_sentence': mkn_segment_sentence_full,
        'eng_segment_sentence': eng_segment_sentence_full,
        'start': start_segment_sentence_full,
        'wav_files': wav_segment_files_full
    })

    df.to_csv(f'jakarta_field_station_with_wav.csv', index=False)
    return df

base_url = "https://archive.mpi.nl/tla/islandora/object/tla%3A1839_00_0000_0000_0022_5AD1_C"
df = main(base_url)