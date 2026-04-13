import requests
import pandas as pd
import time
import os
from datetime import datetime

HANDLE = "vishuyadav.bsky.social"
APP_PASSWORD = "dvzw-p5vh-xbwq-ejl6"

# SEARCH_TERMS = [
#     "SEBI",
#     "SEBI ban",
#     "SEBI rules",
#     "RBI",
#     "RBI rate",
#     "IndiaInvestments",
#     "Indian stock market"
# ]
# OUTPUT_FILE = "data/collected_posts.csv"

SEARCH_TERMS = [
    "ChatGPT",
    "GPT-4",
    "Claude AI",
    "Gemini AI",
    "OpenAI",
    "LLM",
    "artificial intelligence",
    "AI model",
    "machine learning",
    "AI announcement",
    "AI release",
    "AI hype",
    "large language model",
    "AI tools"
]

OUTPUT_FILE = "data/ai_tech_posts.csv"

def login():
    url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    r = requests.post(url, json={
        "identifier": HANDLE,
        "password": APP_PASSWORD
    })
    if r.status_code == 200:
        print("Login successful")
        return r.json()["accessJwt"]
    else:
        print(f"Login failed: {r.status_code}")
        return None

def search_posts(token, query, limit=100):
    headers = {"Authorization": f"Bearer {token}"}
    url = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
    params = {"q": query, "limit": limit}
    
    r = requests.get(url, headers=headers, params=params)
    if r.status_code == 200:
        return r.json().get("posts", [])
    else:
        print(f"Search failed for '{query}': {r.status_code}")
        return []

def extract_post_data(post, search_term):
    try:
        return {
            "post_id": post.get("uri", ""),
            "author": post["author"].get("handle", ""),
            "author_display": post["author"].get("displayName", ""),
            "text": post["record"].get("text", ""),
            "created_at": post["record"].get("createdAt", ""),
            "like_count": post.get("likeCount", 0),
            "reply_count": post.get("replyCount", 0),
            "repost_count": post.get("repostCount", 0),
            "search_term": search_term,
            "collected_at": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error extracting post: {e}")
        return None

def save_posts(posts_data):
    df_new = pd.DataFrame(posts_data)
    
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset=["post_id"], inplace=True)
        df_combined.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved. Total unique posts: {len(df_combined)}")
    else:
        df_new.drop_duplicates(subset=["post_id"], inplace=True)
        df_new.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved. Total unique posts: {len(df_new)}")

def run_scraper():
    print(f"\n{'='*50}")
    print(f"Scraper started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")
    
    token = login()
    if not token:
        return
    
    all_posts = []
    
    for term in SEARCH_TERMS:
        print(f"Searching: '{term}'...")
        posts = search_posts(token, term, limit=100)
        
        for post in posts:
            extracted = extract_post_data(post, term)
            if extracted:
                all_posts.append(extracted)
        
        print(f"  Found {len(posts)} posts")
        time.sleep(2)
    
    if all_posts:
        save_posts(all_posts)
    
    print(f"\nRound complete at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Collected {len(all_posts)} posts this round")

if __name__ == "__main__":
    while True:
        run_scraper()
        print("\nWaiting 15 minutes before next collection...")
        time.sleep(900)