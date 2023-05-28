import configparser
from datetime import datetime, timedelta
import feedparser
import hashlib
import json
import nltk
import os
import psycopg2

# adjusted from a script I already have running elsewhere

def open_file(filename):
    fobj = open(filename, 'w', encoding='utf8', errors='ignore')
    return fobj


def write_file(fobj, line):
    fobj.write(str(line))
    fobj.write("\n")


def close_file(fobj):
    fobj.close()


def get_configs(path="nltk_nounremover.ini"):
    config = configparser.ConfigParser()
    config.read(path)
    return config

# news aggregation first
def get_feed_items(config):
    """ get feed items from news sources, place originals in one db and with nouns removed in another """
    with open("nltk_feedlist.json", "r") as f:
        newsfeedList = json.load(f)
    newTitleList = []
    sep1 = '=' * 75
    sep2 = '-' * 50

    conn, cur = connect_to_db(db=config["db"]["database"])
    conn2, cur2 = connect_to_db(db=config["db"]["database2"])
    for url in newsfeedList:
        newsfeed = feedparser.parse(url)
        entries = newsfeed['entries']
        for item in entries:
            storylink = item['link']
            source = item['title_detail']['base']
            raw = json.dumps(item)
            if item.get('published_parsed'):
                published = datetime(*item['published_parsed'][:6])
            else:
                published = datetime.utcnow()
            # adjust for utc
            if published > datetime.now():
                # set published back 7 hours
                published = published - timedelta(hours=7)
            byte_title = item['title'].encode('utf8')
            checksum = hashlib.sha256(byte_title).hexdigest()
            tokenized = nltk.word_tokenize(item['title'].replace("'", ""))
            # nltk.download('averaged_perceptron_tagger')
            # nltk.download('punkt')
            nouns = [(word, pos) for (word, pos) in nltk.pos_tag(tokenized) if (pos[:2] == 'NN')]
            all = [(word, pos) for (word, pos) in nltk.pos_tag(tokenized)]
            nounz = [(word, pos) for (word, pos) in nltk.pos_tag(tokenized) if
                     (pos == 'NNP') or (pos == 'NNPS')]  # if (pos[:2] == 'NN')]
            newTitle = item['title'].split(':')[1] if ':' in item['title'] else item['title']
            newTitle = newTitle.replace("'", "")
            for word in nounz:
                if word[0] in newTitle:
                    newTitle = newTitle.replace(word[0], '*')
            for word in nouns:
                if word[0] in newTitle:
                    newTitle2 = newTitle.replace(word[0], '*')
            newTitleList.append(newTitle)

            if item.get('summary'):
                fullSummary = item['summary']
            else:
                fullSummary = ""
            relevantSummary = fullSummary.split('<', 1)[0]
            tokenized2 = nltk.word_tokenize(relevantSummary)
            # nltk.download('averaged_perceptron_tagger')
            # nltk.download('punkt')
            nouns2 = [word for (word, pos) in nltk.pos_tag(tokenized2) if (pos[:2] == 'NN')]
            relevantSummary2 = relevantSummary
            for word in nouns2:
                if word in relevantSummary2:
                    relevantSummary2 = relevantSummary2.replace(word, '*')
            write_to_db(cur, conn, 'news_feed', newTitle, relevantSummary2, checksum, storylink[:255], published, raw, source[:255])
            write_to_db(cur2, conn2, 'feeds', item['title'], relevantSummary, checksum, storylink[:255], published, raw, source[:255])
    close_cursor(cur)
    close_conn(conn)
    close_cursor(cur2)
    close_conn(conn2)

# could remove this and just use the one above
def connect_to_db(configFile="nltk_nounremover.ini", db=None):
    config = configparser.ConfigParser()
    config.read(configFile)
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASS")
    host = config["db"]["hostname"]
    conn = psycopg2.connect(dbname=db, user=user, password=password, host=host)
    cur = conn.cursor()
    return conn, cur


def write_to_db(cur, conn, table, headline, story, checksum, link, published, raw, source):
    sql = f"""insert into {table} (headline, story, checksum, storylink, published, fullitem, source) values (%s, %s, %s, %s, %s, %s, %s);"""
    try:
        cur.execute(sql, (headline, story, checksum, link, published, raw, source))
    except psycopg2.errors.UniqueViolation:
        #print(f"Unique Error: Story {headline} is already in db...\n")
        pass
    conn.commit()


def close_cursor(cur):
    cur.close()


def close_conn(conn):
    conn.close()


if __name__ == '__main__':
    get_feed_items()