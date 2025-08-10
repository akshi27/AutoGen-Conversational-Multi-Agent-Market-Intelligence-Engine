import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import time

# Load environment variables from the .env file.
load_dotenv()

# Load Bearer Token from environment variable. This is the API key for X.
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")

# API endpoints for X v2. The 'recent search' endpoint is used for both.
SEARCH_URL = "https://api.x.com/2/tweets/search/recent"

# Separate lists for brands and product types.
BRANDS = [
    "Atomberg", "Crompten", "Havells", "Orient", "Polycab", "LG", "Luminous", "Vguard", "Honeywell", "Polycab", "Bajaj"
]

PRODUCT_TYPES = [
    "smart fan", "bldc fan"
]

def create_headers():
    """
    Creates the necessary headers for the API request, including the Bearer Token.
    Raises a ValueError if the token is not found in the environment variables.
    """
    if not BEARER_TOKEN:
        raise ValueError("X_BEARER_TOKEN is not set in environment variables. "
                         "Please check your .env file and ensure the variable is named correctly.")
    return {"Authorization": f"Bearer {BEARER_TOKEN}"}

def get_top_n_popular_posts(tweets_data, n):
    """
    Sorts a list of tweets by a popularity score and returns the top N.
    The popularity score is a simple sum of likes and retweets.

    Args:
        tweets_data (list): A list of tweet dictionaries from the API response.
        n (int): The number of top posts to return.

    Returns:
        list: A new list containing the top N most popular posts.
    """
    if not tweets_data:
        return []

    # Sort tweets in descending order based on a combined score of likes and retweets.
    sorted_tweets = sorted(
        tweets_data,
        key=lambda t: t['public_metrics'].get('like_count', 0) + t['public_metrics'].get('retweet_count', 0),
        reverse=True
    )
    return sorted_tweets[:n]

def scrape_x_tweets(brands, product_types, max_posts_to_scrape=40, max_replies_per_post=4, save=True):
    """
    Scrapes X for a specified number of popular posts and their replies.

    Args:
        brands (list): A list of brand names to search for.
        product_types (list): A list of product type keywords to search for.
        max_posts_to_scrape (int): The total number of popular posts to scrape.
        max_replies_per_post (int): The maximum number of replies to fetch for each post.
        save (bool): If True, saves the results to a JSON file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a tweet.
    """
    print("[+] Building a more targeted query for X...")
    headers = create_headers()
    all_tweets = []

    # Construct a more targeted query string.
    query_parts = [f'("{brand}" "{product_type}")' for brand in brands for product_type in product_types]
    query_string = " OR ".join(query_parts)
    
    # Add filters to exclude retweets and include only English language posts.
    full_query = f"{query_string} -is:retweet lang:en"

    # Fetch a large batch of tweets to find the most popular ones.
    # The API will return up to 100 posts per request.
    params = {
        "query": full_query,
        "max_results": 100,
        "tweet.fields": "id,text,author_id,created_at,public_metrics,conversation_id",
        "expansions": "author_id",
        "user.fields": "username"
    }
    
    try:
        print("[+] Fetching an initial batch of posts to identify the most popular...")
        response = requests.get(SEARCH_URL, headers=headers, params=params)
        
        if response.status_code == 429:
            print("[!] Rate limit hit: Too many requests. Please wait a few minutes before trying again.")
            return []
        if response.status_code != 200:
            print(f"[!] Error fetching tweets: {response.status_code} {response.text}")
            return []

        response_data = response.json()
        tweets_data = response_data.get("data", [])
        
        # Select only the top 'max_posts_to_scrape' posts based on popularity.
        popular_tweets = get_top_n_popular_posts(tweets_data, max_posts_to_scrape)
        
        if not popular_tweets:
            print("[!] No popular tweets found matching the query.")
            return []

        users_data = {u["id"]: u["username"] for u in response_data.get("includes", {}).get("users", [])}
        
        print(f"[+] Processing the top {len(popular_tweets)} popular posts...")

        for tweet in popular_tweets:
            text_lower = tweet["text"].lower()
            mentioned_brands = [b for b in brands if b in text_lower]
            mentioned_product_types = [p for p in product_types if p in text_lower]
            metrics = tweet.get("public_metrics", {})
            
            tweet_data = {
                "id": tweet["id"],
                "username": users_data.get(tweet["author_id"], ""),
                "date": tweet["created_at"],
                "content": tweet["text"],
                "url": f"https://x.com/{users_data.get(tweet['author_id'], '')}/status/{tweet['id']}",
                "likeCount": metrics.get("like_count", 0),
                "retweetCount": metrics.get("retweet_count", 0),
                "replyCount": metrics.get("reply_count", 0),
                "platform": "X",
                "brands_mentioned": mentioned_brands,
                "product_types_mentioned": mentioned_product_types,
                "replies": []
            }
            
            # --- Fetching Replies for Each Tweet ---
            # This part fetches tweets that are replies to the current conversation.
            reply_params = {
                "query": f"conversation_id:{tweet['conversation_id']} -from:{users_data.get(tweet['author_id'], '')} lang:en",
                "tweet.fields": "id,text,author_id,created_at,public_metrics,in_reply_to_user_id",
                "expansions": "author_id",
                "user.fields": "username",
                "max_results": max_replies_per_post # Limiting the number of replies.
            }
            
            reply_resp = requests.get(SEARCH_URL, headers=headers, params=reply_params)
            if reply_resp.status_code == 200:
                replies_data = reply_resp.json().get("data", [])
                reply_users = {u["id"]: u["username"] for u in reply_resp.json().get("includes", {}).get("users", [])}
                
                for r in replies_data:
                    tweet_data["replies"].append({
                        "username": reply_users.get(r["author_id"], ""),
                        "date": r["created_at"],
                        "content": r["text"],
                        "url": f"https://x.com/{reply_users.get(r['author_id'], '')}/status/{r['id']}"
                    })
            
            all_tweets.append(tweet_data)
            
    except Exception as e:
        print(f"[!] Exception while fetching X tweets: {e}")
        return []

    if save:
        os.makedirs("data/raw", exist_ok=True)
        filename = f"data/raw/x_fan_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_tweets, f, indent=2, ensure_ascii=False)
        print(f"[✓] Saved X results to {filename}")

    return all_tweets

if __name__ == "__main__":
    brands_to_scrape = [
        "atomberg", "crompten", "havells", "orient", "polycab", "lg", "luminous", "vguard", "honeywell", "polycab", "bajaj"
    ]


    product_types_to_scrape = [
        "smart fan", "bldc fan"
    ]
    
    # Scrape the top 40 posts and 4 replies per post.
    results = scrape_x_tweets(brands_to_scrape, product_types_to_scrape, max_posts_to_scrape=40, max_replies_per_post=4)
    
    print("\n--- Scraper Summary ---")
    print(f"Total tweets fetched: {len(results)}")
    
    if results:
        print("\n--- Sample Tweet ---")
        sample_tweet = results[0]
        print(f"Username: {sample_tweet['username']}")
        print(f"Content: {sample_tweet['content'][:100]}...")
        print(f"URL: {sample_tweet['url']}")
        print(f"Brands Mentioned: {', '.join(sample_tweet['brands_mentioned'])}")
        print(f"Product Types Mentioned: {', '.join(sample_tweet['product_types_mentioned'])}")
        print(f"Reply Count: {len(sample_tweet.get('replies', []))}")
