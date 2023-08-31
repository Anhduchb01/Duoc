def get_time_article(article):
    ''' Get year of article '''
    year = article['timeCreatePostOrigin']
    if (year == ""):
        year = article['timeCrawlPage']
    return year
