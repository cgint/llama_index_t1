import requests
from pathlib import Path
from llama_index.readers.schema.base import Document
from llama_index import download_loader
import pickle
import hashlib

load_documents_file_cache_dir = "/data/load_documents-"

request_headers = {
  'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
}

def load_documents(urls, urlPostfix, store_path):
  # https://docs.llamaindex.ai/en/stable/api_reference/readers.html
  # WikipediaReader = download_loader("WikipediaReader")
  # loader = WikipediaReader(urls=urls)
  # documents = loader.load_data(pages=['Guardians of the Galaxy Vol. 3'], auto_suggest=False)

  # UnstructuredURLLoader = download_loader("UnstructuredURLLoader")
  # loader = UnstructuredURLLoader()
  # documents = loader.load_data()

# simply download urls to array of strings using simple python url fetching
  paramHash = hashlib.sha256((str(tuple(urls), urlPostfix)).encode()).hexdigest()
  cache_filename = load_documents_file_cache_dir + "base-" + str(paramHash) + ".pkl"
  print(f"cache_filename: {cache_filename}")
  if Path(cache_filename).exists():
    print(f"Loading documents from cache file {cache_filename} ...")
    with open(cache_filename, "rb") as f:
      loaded_documents = pickle.load(f)
    return loaded_documents
  
  documents = []
  MarkdownReader = download_loader("MarkdownReader")
  mdLoader = MarkdownReader()
  for dlUrl in urls:
      final_url = dlUrl + urlPostfix
      content = requests.get(final_url, headers=request_headers).text
      # extract first healine starting with # ending with newline
      firstHeadStart = content.find("#")
      firstHeadline = content[firstHeadStart:content.find("\n", firstHeadStart)].replace("#", "").strip()
      extra_info = {"Mostbauer Eintrag": firstHeadline} # "url": dlUrl
      print(f"Fetching as document from {final_url} ...")
      # documents.append(Document(text=content, extra_info=extra_info))
      documents.extend(mdLoader.load_data(
        file=Path('.'), 
        content=content,
        extra_info=extra_info
        )
      )
  # save documents to disc as text
  _store_documents(documents, store_path)

  # cache documents on disc using pickle
  with open(cache_filename, "wb") as f:
    pickle.dump(documents, f)

  return documents

def _store_documents(documents, store_path):
  # save documents to disc
  with open(store_path, "w") as f:
    for document in documents:
      f.write(document.text)
      f.write("\n\n\n------------------------------------\n\n\n")


def load_documents_wikipedia(pages, store_path):
  paramHash = hashlib.sha256((str(tuple(pages))).encode()).hexdigest()
  cache_filename = load_documents_file_cache_dir + "wikipedia-" + str(paramHash) + ".pkl"
  print(f"cache_filename: {cache_filename}")
  if Path(cache_filename).exists():
    print(f"Loading documents from cache file {cache_filename} ...")
    with open(cache_filename, "rb") as f:
      loaded_documents = pickle.load(f)
    return loaded_documents

  print(f"About to load documents from wikipedia. {pages} ...")
  # https://docs.llamaindex.ai/en/stable/api_reference/readers.html
  WikipediaReader = download_loader("WikipediaReader")
  loader = WikipediaReader()
  documents = loader.load_data(pages=pages, auto_suggest=False)

  # save documents to disc as text
  _store_documents(documents, store_path)

  # cache documents on disc using pickle
  with open(cache_filename, "wb") as f:
    pickle.dump(documents, f)
  
  return documents

def load_documents_stocknews(stocks, store_path):
  YAHOO_URL = 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=%s&region=US&lang=en-US'
  paramHash = "-".join(stocks)
  cache_filename = load_documents_file_cache_dir + "stocknews-" + str(paramHash) + ".pkl"
  print(f"cache_filename: {cache_filename}")
  if Path(cache_filename).exists():
    print(f"Loading documents from cache file {cache_filename} ...")
    with open(cache_filename, "rb") as f:
      loaded_documents = pickle.load(f)
    return loaded_documents

  print(f"About to load documents from stocknews. {stocks} ...")
  
  df = fetch_rss_feed(stocks, url_pattern=YAHOO_URL, with_content=True, with_sentiment=False)
  # print the dataframe
  print(df)
  
  documents = []
  for index, row in df.iterrows():
    # read contents from row["link"] into string variable
    
    documents.append(Document(
      text=
        "Title:" + row["title"] + "\n" + 
        "Summary:" + row["summary"] + "\n\n" + 
        #"Sentiment-Value of title is " + str(row["sentiment_title"]) + "\n" + 
        #"Sentiment-Value of summary is " + str(row["sentiment_summary"]) + "\n" + 
        #"Sentiment-Value of following content is " + str(row["sentiment_link_target_content"]) + "\n" + 
        "Content" + row["link_target_content"] + "\n\n" +
        "Published: " + row["published"],
      extra_info={
        # "guid": row["guid"],
        #"sentiment_summary": row["sentiment_summary"],
        #"sentiment_title": row["sentiment_title"],
        #"sentiment_link_target_content": row["sentiment_link_target_content"],
        #"published": row["published"],
        # "p_date": row["p_date"],
        # "link": row["link"]
      }))
  
  # save documents to disc as csv
  df.to_csv(store_path.replace(".txt", ".csv"))

  # cache documents on disc using pickle
  with open(cache_filename, "wb") as f:
    pickle.dump(documents, f)
  
  return documents

def fetch_rss_feed(stocks, url_pattern, with_content=False, with_sentiment=False):
  import feedparser
  import nltk
  from nltk.sentiment.vader import SentimentIntensityAnalyzer
  import pandas
  import datetime as dt
  df = pandas.DataFrame(
      columns=['guid',
                'link',
                'stock',
                'title',
                'summary',
                'link_target_content',
                'published',
                'p_date',
                'sentiment_title',
                'sentiment_summary',
                'sentiment_link_target_content']
  )

  """Download VADER"""
  try:
      nltk.data.find('vader_lexicon')
  except LookupError:
      nltk.download('vader_lexicon', quiet=True)

  for stock in stocks:

      """Init new Parser"""
      feed = feedparser.parse(url_pattern % stock)

      for entry in feed.entries:

          """Find guid and skip if exists"""
          guid = df.loc[df['guid'] == entry.guid]
          if len(guid) > 0:
              continue

          """Analyze the sentiment"""
          if with_sentiment:
            sia = SentimentIntensityAnalyzer()
            sentiment_summary = sia.polarity_scores(entry.summary)['compound']
            sentiment_title = sia.polarity_scores(entry.title)['compound']
          else:
            sentiment_summary = None
            sentiment_title = None
 
          """Parse the date"""
          p_date = '%s_%s' % (
              stock, dt.datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S +0000').strftime("%Y-%m-%d"))
          
          if with_content:
            print(f"Fetching link_target_content from {entry.link} ...")
            link_target_content = fetch_html_content(entry.link)
            if with_sentiment:
              sentiment_link_target_content = sia.polarity_scores(link_target_content)['compound']
            else:
              sentiment_link_target_content = None
          else:
            link_target_content = "-link_target_content-download-disabled"

          """Add new entry to DF"""
          row = [
              entry.guid,
              entry.link,
              stock,
              entry.title,
              entry.summary,
              link_target_content,
              entry.published,
              p_date,
              sentiment_title,
              sentiment_summary,
              sentiment_link_target_content
          ]
          df.loc[len(df)] = row

  return df

def fetch_html_content(link):
  from bs4 import BeautifulSoup
  import requests
  plain_html = requests.get(link, headers=request_headers).text
  soup = BeautifulSoup(plain_html, 'html.parser')
  # fetch body only
  content_part = soup.find('article')
  if content_part is None:
    content_part = soup.find('body')
  # remove script and style tags
  for tagsToRemove in content_part(["script", "style"]):
    tagsToRemove.extract()
  plain_text = content_part.get_text()
  if "Oops!Something went wrong." in plain_text or "Please try again later." in plain_text:
    print("Text for link " + link + " contains error message 'Oops!Something went wrong.' or 'Please try again later.'. Removing ...")
    plain_text = plain_text.replace("Oops!Something went wrong.", "").replace("Please try again later.", "")

  return plain_text


def load_documents_from_files_in_zipfile(zipfile_path, store_path):
  from zipfile import ZipFile

  paramHash = hashlib.sha256((str(zipfile_path)).encode()).hexdigest()
  cache_filename = load_documents_file_cache_dir + "zipfile-" + str(paramHash) + ".pkl"
  print(f"cache_filename: {cache_filename}")
  if Path(cache_filename).exists():
    print(f"Loading documents from cache file {cache_filename} ...")
    with open(cache_filename, "rb") as f:
      loaded_documents = pickle.load(f)
    return loaded_documents

  print(f"About to load documents from zipfile. {zipfile_path} ...")
  # read contents from zip file using python libraries
  documents = []
  with ZipFile(zipfile_path, 'r') as zip:
    # printing all the contents of the zip file
    #zip.printdir()
    # extracting all the files
    print('Extracting all the files now...')
    #zip.extractall()
    print('Done!')
    for file in zip.namelist():
      print(f"Reading file {file} from zip ...")
      with zip.open(file) as f:
        content = f.read()
        documents.append(Document(text=content, extra_info={"filename": file}))
  # save documents to disc as text
  _store_documents(documents, store_path)

  # cache documents on disc using pickle
  with open(cache_filename, "wb") as f:
    pickle.dump(documents, f)
  
  return documents