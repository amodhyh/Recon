from typing import Annotated

import feedparser
from pydantic import Field
from urllib.parse import quote_plus
from langchain_core.tools import tool
from recon.models import Article,Company
from email.utils import parsedate_to_datetime
import logging


rss_logger=logging.getLogger(name="rss-logger")

@tool   
async def fetch_feed(
    company: Annotated[Company, Field(description="Company to search recent news for")],
    days_before: Annotated[
        int, Field(ge=1, le=30, description="Only include articles from the last N days")
    ] = 3,
    limit: Annotated[
        int, Field(ge=1, le=50, description="Maximum number of articles to return")
    ] = 10,
) -> list[Article]:
    """Search Google News for recent articles about a company.

    Returns a list of Article objects (title, url, published date) for the
    given company, limited to the last `days_before` days and at most `limit` results.
    """
    
    q = quote_plus(f'"{company.company_name}" when:{days_before}d')
    feed_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"   
    article_list: list[Article] = []    
    
    d = feedparser.parse(feed_url)
    if d.bozo:                                                                              
        bozo_exc = d.get("bozo_exception")                                                         
        rss_logger.warning(f"Feed parse issue: {bozo_exc}")        

    for art_heading in d.entries[:limit]:
        try:
            pub = art_heading.get("published")
            article = Article(
                competitor = company,
                title = art_heading.title,
                url = art_heading.link,
                summary = "",
                published=parsedate_to_datetime(pub) if pub else None
                )
            
            article_list.append(article)
        except Exception as e:
            rss_logger.exception(e)
         
    return article_list

    

def main():
    print(fetch_feed.invoke({"company": {"company_name": "Anthropic"}}))
   
  

if __name__=="__main__":
    main()