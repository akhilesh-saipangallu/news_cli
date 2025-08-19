# NEWS CLI (POV)

A tiny CLI to demo **anonymous user personalization** for a news homepage using Redis (RedisSearch + RedisJSON).

---

## Prerequisites

* Redis with **RedisSearch** and **RedisJSON** modules
* Python 3.9+

---

## Quick start

1. **Init and load data**

   ```bash
   python app.py init
   ```

   Re-init and **clear existing users**:

   ```bash
   python app.py init delete_users=true
   ```

2. **Get headlines (creates a new anon user if uid not provided)**

   ```bash
   python app.py headlines
   # prints created user id and lists top headlines with topics
   ```

3. **View an article (track a click/view)**

   ```bash
   python app.py view aid=349 uid=<uid>
   ```

4. **Personalized headlines**

   ```bash
   python app.py personal_headlines uid=<uid>
   ```

---

## Commands (cheat sheet)

* `python app.py init`

  * Creates Redis indices and loads demo articles
* `python app.py init delete_users=true`

  * Same as above, but deletes existing users before loading
* `python app.py headlines`

  * Returns **top 10** headlines by popularity and **creates a new user** (pass `n=15` for top 15 news articles)
* `python app.py headlines uid=46fe25ff`

  * Returns **top 10** headlines by popularity for an **existing user** (no new user created)
* `python app.py view aid=349 uid=46fe25ff`

  * Records that the user viewed/clicked the article; updates their topic counts and history
* `python app.py personal_headlines uid=46fe25ff`

  * Returns a **personalized** list (mostly fresh & popular, with a nudge toward the user’s topics)
* `python app.py v_search search_text='football' topic='sports'`

  * Performs a VSS based by converting the search-text on the article topic subset

> Default page size is 10; pass `n=15` to get 15 items where supported.

---

## What “gradual” means (in this demo)

* New / low-activity users: **pure popularity**
* After a few clicks: \~**70%** fresh & popular + \~**30%** from the user’s top topics
* Safety rails: per-topic cap, no repeats of already clicked articles

---

## Typical demo flow

```bash
python app.py init
python app.py headlines            # creates a user and shows base feed
python app.py view aid=349 uid=<uid>
python app.py headlines uid=<uid>  # slight nudge
python app.py view aid=221 uid=<uid>
python app.py personal_headlines uid=<uid>  # clearer personalization, still mixed
```

---

## Notes

* Configure Redis via `REDIS_URL` env var.
* Articles are stored as JSON and indexed for fast querying.
* Users are anonymous: identified only by a random `uid` stored client-side (cookie in real usage).
