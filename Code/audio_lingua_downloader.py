from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

# Set main write path
main_path = Path().resolve().parent.joinpath('Data/Audio_Lingua')
main_path.mkdir(exist_ok=True)
# Get main page
main_request = requests.get('https://www.audio-lingua.eu/?lang=en')
soup = BeautifulSoup(main_request.content)
# Find all links with "Latest update" in title
links = soup.find_all('a', title=re.compile("Latest update"))
# Get links to all language pages
language_pages = ['https://www.audio-lingua.eu/' + link.get('href') for link in links]
for language_page in language_pages[1:]:
    all_filenames = []
    all_authors = []
    # Get first page for current language
    current_request = requests.get(language_page)
    current_soup = BeautifulSoup(current_request.content)
    # Get language name and set language path
    language = current_soup.find_all('h1')[1].string
    language_path = main_path.joinpath(language)
    language_path.mkdir()
    # Finds navigation links
    nav_links = [i.get('href') 
                for i in current_soup.find_all('a', href=re.compile('pagination_articles'))]
    # Get the last pagination number for looping
    pagination = int([re.split('=|#', i)[1] for i in nav_links][-1])
    # Loop over pagination number in increments of 5
    for page_number in range(0, pagination+1, 5):
        print(f"Downloading {language} page {page_number}")
        page_link = ''.join([language_page, '&debut_articles=', 
                             str(page_number),'#pagination_articles'])
        current_request = requests.get(page_link)
        current_soup = BeautifulSoup(current_request.content)
        # Find all audio links
        current_links = ['https://www.audio-lingua.eu/' + link.get('src') 
                        for link in current_soup.find_all('source')]
        # Get filenames and authors (only takes first author, possible multiple authors)
        current_filenames = [link.split('/')[-1] for link in current_links]
        current_authors = [i.find_all('span')[0].string 
                           for i in current_soup.find_all('span', class_='authors')]
        # Skip iteration if there is a file with no author
        if len(current_authors) != len(current_filenames):
            continue
        # Update list of filenames and authors
        all_filenames += current_filenames
        all_authors += current_authors
        # Open audio files and write to language path
        for filename, link in zip(current_filenames, current_links):
            file_url = requests.get(link)
            with open(language_path.joinpath(filename), 'wb') as file:
                file.write(file_url.content)
    # Write filenames and authors to csv when done with loop
    authors_df = pd.DataFrame({'author': all_authors, 'filename': all_filenames})
    authors_df.to_csv(main_path.joinpath(language + '.csv'), index=False)