import argparse
import sys
import time
import uuid
import json
import random
import datetime

from redis import Redis
from redisvl.schema import IndexSchema
from redisvl.index import SearchIndex
from redisvl.query import FilterQuery, VectorQuery
from redisvl.query.filter import Tag, Num
from redis.commands.search.aggregation import Desc

import numpy as np
from sentence_transformers import SentenceTransformer


REDIS_URL = 'redis://localhost:6379/0'
ARTICLES_PATH = 'articles_500.json'
# prefixes
ARTICLE_PREFIX = 'ht:article:'
USER_PREFIX = 'ht:user:'
# indexes
ARTICLE_INDEX = 'ht:idx:articles'
USER_INDEX = 'ht:idx:users'
# TTL for anonymous users (14 days in seconds)
USER_TTL = 14 * 24 * 3600

em_model = SentenceTransformer('all-MiniLM-L6-v2')


def redis_client():
    return Redis.from_url(REDIS_URL, decode_responses=True)


def print_separator(tag):
    s, x = '=', 15
    print(s*x, tag, s*x)


def print_with_separator(tag, *args):
    s, x = '=', 15
    print(s*x, tag, s*x)
    print(*args)
    print(s*x, tag, s*x)


def generate_random_unix_timestamp():
    now = datetime.datetime.now()
    twenty_four_hours_ago = now - datetime.timedelta(hours=24)

    unix_now = int(time.mktime(now.timetuple()))
    unix_twenty_four_hours_ago = int(time.mktime(twenty_four_hours_ago.timetuple()))

    random_timestamp = random.randint(unix_twenty_four_hours_ago, unix_now)

    return random_timestamp


def article_index_schema():
    return IndexSchema.from_dict({
        'index': {
            'name': ARTICLE_INDEX,
            'prefix': ARTICLE_PREFIX,
            'storage_type': 'json',
        },
        'fields': [
            {'name': 'storyId', 'path': '$.storyId', 'type': 'tag'},
            {'name': 'headline', 'path': '$.headline', 'type': 'text', 'attrs': {'sortable': False}},
            {'name': 'summary', 'path': '$.summary', 'type': 'text', 'attrs': {'sortable': False}},
            {'name': 'keywords', 'path': '$.keywords[*]', 'type': 'tag', 'attrs': {'separator': ','}},
            {'name': 'topics', 'path': '$.topics[*]', 'type': 'tag', 'attrs': {'separator': ','}},
            {'name': 'created_at', 'path': '$.created_at', 'type': 'numeric', 'attrs': {'sortable': True}},
            {'name': 'pop_score', 'path': '$.pop_score', 'type': 'numeric', 'attrs': {'sortable': True}},
            {
                'name': 'headline_embedding',
                'path': '$.headline_embedding',
                'type': 'vector',
                'attrs': {
                    'algorithm': 'HNSW', # or 'FLAT'
                    'datatype': 'FLOAT32',
                    'dims': 384,
                    'distance_metric': 'COSINE',
                }
            }
        ],
    })


def user_index_schema():
    return IndexSchema.from_dict({
        'index': {
            'name': USER_INDEX,
            'prefix': USER_PREFIX,
            'storage_type': 'json',
        },
        'fields':[
            {'name': 'id', 'path': '$.id', 'type': 'text'},
            {'name': 'topic_name', 'path': '$.topics[*].name', 'type': 'tag', 'attrs': {'separator': ','}},
            {'name': 'topic_count', 'path': '$.topics[*].count', 'type': 'numeric', 'attrs': {'sortable': True}},
            {'name': 'clicked', 'path': '$.clicked[*]', 'type': 'tag', 'attrs': {'separator': ','}},
        ],
    })


def load_articles_to_redis():
    r = redis_client()

    with open(ARTICLES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError('Input JSON must be a list of article objects')

    def _rand_pop():
        x = random.uniform(0, 1)
        return float(f'{x:.4f}')

    pipe = r.pipeline(transaction=False)
    now = int(time.time())

    for art in data:
        story_id = str(art.get('storyId', '')).strip()
        if not story_id:
            continue

        headline = (art.get('headline') or '').strip()
        summary = (art.get('summary') or '').strip()

        keywords = art.get('keywords') or []
        keywords = [str(k).lower().replace(' ', '-') for k in keywords]

        topics = art.get('topics') or []
        topics = [str(t).lower().replace(' ', '-') for t in topics][:3]

        created = generate_random_unix_timestamp()

        eligible = art.get('eligible', True)
        eligible_tag = 'true' if bool(eligible) else 'false'  # TAG-friendly

        headline_embedding = embed_sentence(headline)

        obj = {
            'storyId': story_id,
            'headline': headline,
            'summary': summary,
            'keywords': keywords,
            'topics': topics,
            'created_at': created,
            'pop_score': _rand_pop(),     # <-- randomized here
            'eligible': eligible_tag,
            'headline_embedding': headline_embedding.tolist(),
        }

        pipe.json().set(f'{ARTICLE_PREFIX}{story_id}', '$', obj)

    pipe.execute()


def delete_all_users():
    r = redis_client()
    for key in r.scan_iter(f'{USER_PREFIX}*'):
        r.delete(key)


def delete_all_articles():
    r = redis_client()
    for key in r.scan_iter(f'{ARTICLE_PREFIX}*'):
        r.delete(key)


def action_init(**kwargs):
    """
    Kwargs:
        delete_users    : if true delete all users
        delete_articles : if true delete all articles
    """

    r = redis_client()

    # Articles index
    art_idx = SearchIndex(article_index_schema(), r)
    if not art_idx.exists():
        art_idx.create()
        print(f'Created index: {ARTICLE_INDEX}')
    else:
        print(f'Index exists: {ARTICLE_INDEX}')

    # Users index
    usr_idx = SearchIndex(user_index_schema(), r)
    if not usr_idx.exists():
        usr_idx.create()
        print(f'Created index: {USER_INDEX}')
    else:
        print(f'Index exists: {USER_INDEX}')

    delete_users = kwargs.get('delete_users')
    if delete_users and delete_users == 'true':
        delete_all_users()

    delete_articles = kwargs.get('delete_articles')
    if delete_articles and delete_articles == 'true':
        delete_all_articles()

    load_articles_to_redis()
    print('init: done')


def create_new_user():
    uid = str(uuid.uuid4())[:8]
    key = f'{USER_PREFIX}{uid}'

    doc = {'id': uid}

    r = redis_client()
    r.json().set(key, '$', doc)
    r.expire(key, USER_TTL)

    print_with_separator('info', 'new user created:', uid)


def print_headlines(data):
    if not data or len(data) == 0:
        print('No results found')
        return

    f_data = []
    for i, each in enumerate(data):
        t = {
            'id': i + 1,
            'storyId': each['storyId'],
            'topics': each['topics'],
            'headline': each['headline'],
        }
        if 'vector_distance' in each:
            t['vector_distance'] = each['vector_distance']

        f_data.append(t)
    data = f_data

    if not isinstance(data, list) or not data:
        print('Error: JSON must represent a non-empty list of objects.')
        return

    # Extract headers from the keys of the first dictionary
    headers = list(data[0].keys())

    # Calculate the maximum width for each column dynamically
    column_widths = {header: len(header) for header in headers}
    for row in data:
        for header in headers:
            column_widths[header] = max(column_widths[header], len(str(row.get(header, ''))))

    # Print the header row
    header_line = ''
    for header in headers:
        header_line += f'{header:<{column_widths[header] + 3}}'
    print(header_line.strip())
    print('-' * len(header_line))

    # Print each data row
    for row in data:
        row_line = ''
        for header in headers:
            value = str(row.get(header, ''))
            row_line += f'{value:<{column_widths[header] + 3}}'
        print(row_line.strip())


def action_headlines(**kwargs):
    """
    Kwargs:
        uid : user id (e.g., 'a9b0d6d3')
        n   : number of headlines; defaults to 10
    """

    uid = kwargs.get('uid')
    if not uid:
        create_new_user()

    n = int(kwargs.get('n', 10))
    r = redis_client()

    min_epoch = int(time.time()) - 24 * 3600

    idx = SearchIndex(article_index_schema(), redis_client())
    res = idx.query(
        FilterQuery(
            return_fields=['storyId', 'topics', 'headline'],
            filter_expression=(
                Num('created_at').between(min_epoch, '+inf')
            ),
            num_results=n,
        ).sort_by('pop_score', asc=False).dialect(3)
    )

    print_headlines(res)


def action_view(**kwargs):
    """
    Record that a user viewed/clicked an article and update their profile.

    Kwargs (required):
        uid: user id (e.g., 'a9b0d6d3')
        aid: article/story id (e.g., '133')

    User JSON shape:
    {
        'id': 'a9b0d6d3',
        'topics': [{'name': 'bollywood', 'count': 5}, {'name': 'science', 'count': 15}],
        'clicked': [1, 15, 99]
    }
    """

    uid = kwargs.get('uid')
    aid = kwargs.get('aid')
    if not uid or not aid:
        print_with_separator('error', 'provide uid and aid')
        return

    r = redis_client()
    u_key = f'{USER_PREFIX}{uid}'
    a_key = f'{ARTICLE_PREFIX}{aid}'

    if not r.exists(u_key):
        print_with_separator('error', f'user not found: {uid}')
        return
    if not r.exists(a_key):
        print_with_separator('error', f'article not found: {aid}')
        return

    # Get article topics + headline
    article = r.json().get(a_key) or {}

    a_topics = article.get('topics', [])

    # Fetch user doc (entire object so we can safely mutate arrays)
    u_doc = r.json().get(u_key) or {}

    # 1) Update clicked (de-dupe)
    clicked = u_doc.get('clicked') or []
    aid_str = str(aid)
    if aid_str in clicked:
        print_with_separator('info', f'article {aid} already viewed by {uid}')
        return

    clicked.append(aid_str)
    clicked = list(set(clicked))

    # 2) Update topics counts
    # sample: [{'name': 'bollywood', 'count': 5}, {'name': 'science', 'count': 15}]
    u_topics = u_doc.get('topics') or []
    t_map = {}
    for item in u_topics:
        t_map[item['name'].lower()] = item['count']

    for a_topic in a_topics:
        if a_topic.lower() in t_map:
            t_map[a_topic.lower()] += 1
        else:
            t_map[a_topic.lower()] = 1

    updated_u_topics = []
    for k, v in t_map.items():
        updated_u_topics.append({'name': k, 'count': v})

    u_doc['clicked'] = clicked
    u_doc['topics'] = updated_u_topics
    r.json().set(u_key, '$', u_doc)

    print_separator('info')
    print(article.get('headline', ''))
    print('topics:', a_topics)
    print(f'article {aid} viewed by {uid}')
    print_separator('info')


def get_top_2_topics(u_doc):
    """
    u_doc:
    {
        'id': 'a9b0d6d3',
        'topics': [{'name': 'bollywood', 'count': 5}, {'name': 'science', 'count': 15}],
        'clicked': [1, 15, 99]
    }
    """

    u_topics = u_doc.get('topics', [])
    if not u_topics:
        return []

    sorted_data = sorted(u_topics, key=lambda x: x['count'], reverse=True)
    result = []
    for i, each in enumerate(sorted_data):
        if i > 1:
            break
        result.append(each['name'])

    return result


def action_personalized_headlines(**kwargs):
    """
    Kwargs (required):
        uid : user id (e.g., 'a9b0d6d3')
        n   : number of headlines; defaults to 10
    """

    uid = kwargs.get('uid')
    if not uid:
        print_with_separator('error', 'provide uid(user id)')
        return

    r = redis_client()
    u_key = f'{USER_PREFIX}{uid}'

    u_doc = r.json().get(u_key) or {}
    if not u_doc:
        print_with_separator('error', f'user not found: {uid}')
        return

    n = int(kwargs.get('n', 10))
    headline_n = int(n * 0.5)
    personal_n = n - headline_n
    min_epoch = int(time.time()) - 24 * 3600

    idx = SearchIndex(article_index_schema(), redis_client())
    headline_res = idx.query(
        FilterQuery(
            return_fields=['storyId', 'topics', 'headline'],
            filter_expression=(
                Num('created_at').between(min_epoch, '+inf')
            ),
            num_results=headline_n,
        ).sort_by('pop_score', asc=False).dialect(3)
    )

    filter_expression = (Num('created_at').between(min_epoch, '+inf'))

    top_2_topics = get_top_2_topics(u_doc)
    print('user interests:', top_2_topics)
    if top_2_topics:
        filter_expression = filter_expression & (Tag('topics') == top_2_topics)

    exclude_articles = [str(each['storyId']) for each in headline_res]
    filter_expression = filter_expression & ((Tag('storyId') != exclude_articles))

    personal_res = idx.query(
        FilterQuery(
            return_fields=['storyId', 'topics', 'headline'],
            filter_expression=filter_expression,
            num_results=personal_n,
        ).sort_by('pop_score', asc=False).dialect(3)
    )

    res = headline_res
    res.extend(personal_res)
    print_headlines(res)


def parse_kv_args(kv_list):
    out = {}
    for item in kv_list:
        if '=' not in item:
            continue
        k, v = item.split('=', 1)
        # naive cast to int/float if possible
        if v.isdigit():
            out[k] = int(v)
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def embed_sentence(sentence: str):
    """Return a 384-dim semantic embedding for an English sentence."""
    v = em_model.encode(sentence, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)



def action_v_search(**kwargs):
    """
    Kwargs (required):
        search_text : search text to perform VSS on

    Kwargs (optional):
        topic       : topic to filter on
        n           : number of headlines; defaults to 10
    """

    search_text = kwargs.get('search_text')
    if not search_text:
        print_with_separator('error', 'provide search_text')
        return
    search_emb = embed_sentence(search_text)

    topic = kwargs.get('topic')
    filter_expression = None
    if topic:
        filter_expression = (Tag('topics') == topic)

    n = int(kwargs.get('n', 10))

    idx = SearchIndex(article_index_schema(), redis_client())
    res = idx.query(
        VectorQuery(
            vector=search_emb,
            vector_field_name='headline_embedding',
            num_results=n,
            return_fields=['storyId', 'topics', 'headline'],
            return_score=True,
            filter_expression=filter_expression,
            dtype='float32',
            sort_by='vector_distance',
        ).sort_by('vector_distance', asc=True).dialect(3)
    )
    print_headlines(res)


def main():
    parser = argparse.ArgumentParser(description='NewsCLI')
    parser.add_argument(
        'action',
        choices=['init', 'new_user', 'headlines', 'view', 'personal_headlines', 'v_search'],
        help='Action to run'
    )
    parser.add_argument('params', nargs='*', help='Optional key=value params for the action')
    args = parser.parse_args()

    kwargs = parse_kv_args(args.params)

    if args.action == 'init':
        action_init(**kwargs)
    elif args.action == 'headlines':
        action_headlines(**kwargs)
    elif args.action == 'view':
        action_view(**kwargs)
    elif args.action == 'personal_headlines':
        action_personalized_headlines(**kwargs)
    elif args.action == 'v_search':
        action_v_search(**kwargs)
    else:
        print('unknown action', file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()
