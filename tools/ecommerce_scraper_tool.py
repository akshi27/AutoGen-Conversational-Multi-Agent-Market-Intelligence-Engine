import os
import requests
from serpapi import GoogleSearch
import praw
from dotenv import load_dotenv
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
import time
import re

load_dotenv()

# --- Utility Functions ---

def save_data_to_json(data, filename_prefix="feedback_smart_fan"):
    """Save scraped data to a JSON file with timestamp."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = os.path.join(data_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[✓] Data saved to: {filepath}")
        print(f"[✓] Total records saved: {len(data)}")
        return filepath
    except Exception as e:
        print(f"[Error] Failed to save data: {e}")
        return None

def get_serpapi_key():
    return os.getenv("SERP_API_KEY")

def get_reddit_client():
    try:
        return praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
    except Exception as e:
        print(f"[Error] Failed to initialize Reddit client: {e}")
        return None

def post_matches_filters(title, snippet, brands, product_types):
    """Check if post matches brand and product filters."""
    title = title.lower() if title else ""
    snippet = snippet.lower() if snippet else ""
    full_text = f"{title} {snippet}".strip()
    if not full_text:
        return False

    has_brand = any(brand.lower() in full_text for brand in brands)
    has_product = any(ptype.lower() in full_text for ptype in product_types)

    return has_brand and has_product

# --- Reddit Scraper (Consolidated) ---

def scrape_reddit(subreddits, brands, product_types, limit=50, query_mode="broad"):
    """
    Scrapes Reddit using a single, flexible function.
    `query_mode` can be "broad" ("fan") or "targeted" ("brand" "ptype").
    """
    reddit = get_reddit_client()
    if not reddit:
        return []
    
    results = []
    
    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            
            if query_mode == "broad":
                search_queries = ["fan"]
            elif query_mode == "targeted":
                search_queries = [f'"{brand}" "{ptype}"' for brand, ptype in product(brands, product_types)]
            else:
                print(f"[Error] Invalid query_mode: {query_mode}")
                return []
            
            for query in search_queries:
                print(f"[+] Searching r/{sub} for query: '{query}'...")
                
                try:
                    for post in subreddit.search(query, limit=limit, time_filter="year"):
                        if query_mode == "broad" and not post_matches_filters(post.title, post.selftext, brands, product_types):
                            continue
                        
                        comments = []
                        try:
                            post.comments.replace_more(limit=0)
                            for comment in post.comments.list()[:20]:
                                if hasattr(comment, 'body') and comment.body != '[deleted]':
                                    comments.append(comment.body)
                        except Exception as e:
                            print(f"    [Warning] Error fetching comments for post {post.id}: {e}")
                        
                        results.append({
                            "title": post.title,
                            "content": post.selftext,
                            "url": f"https://www.reddit.com{post.permalink}",
                            "score": post.score,
                            "num_comments": post.num_comments,
                            "created_utc": post.created_utc,
                            "platform": "Reddit",
                            "subreddit": sub,
                            "comments": comments[:10],
                            "source": f"reddit_{query_mode}_search",
                            "search_query": query
                        })
                except Exception as e:
                    print(f"    [Warning] Search failed for '{query}': {e}")
                    continue
            
            time.sleep(1) # Add delay between subreddits
        except Exception as e:
            print(f"[Error] Failed to scrape r/{sub}: {e}")
            continue
    
    print(f"[✓] Found {len(results)} {query_mode} Reddit posts.")
    return results

# --- Amazon Scraper (Enhanced) ---

def scrape_amazon_product(product_name):
    serpapi_key = get_serpapi_key()
    if not serpapi_key:
        print("[Warning] SERP_API_KEY not found for Amazon scraping.")
        return []

    print(f"[+] Scraping Amazon for '{product_name}'...")
    try:
        # Search for products on Amazon
        search_params = {
            "engine": "amazon",
            "k": product_name,
            "amazon_domain": "amazon.com",
            "api_key": serpapi_key
        }
        search = GoogleSearch(search_params)
        search_results = search.get_dict()
        
        organic_results = search_results.get("organic_results", [])
        if not organic_results:
            print(f"    [Warning] No Amazon results found for '{product_name}'")
            return []
        
        # Process multiple products (top 3 results)
        amazon_products = []
        for i, product in enumerate(organic_results[:3]):
            asin = product.get("asin")
            if not asin:
                continue
            
            try:
                # Get detailed product information
                product_params = {
                    "engine": "amazon_product", 
                    "asin": asin, 
                    "api_key": serpapi_key
                }
                product_search = GoogleSearch(product_params)
                product_info = product_search.get_dict()
                
                product_results = product_info.get("product_results", {})
                reviews_summary = product_info.get("reviews_summary", {})
                
                # Extract AI-generated customer summary (Customers Say section)
                customers_say = "N/A"
                if "customers_say" in product_info:
                    customers_say_section = product_info.get("customers_say", {})
                    if isinstance(customers_say_section, dict):
                        customers_say = customers_say_section.get("summary", "N/A")
                    elif isinstance(customers_say_section, str):
                        customers_say = customers_say_section
                elif "product_information" in product_info:
                    # Alternative location for customer insights
                    product_info_section = product_info.get("product_information", {})
                    if "customers_say" in product_info_section:
                        customers_say = product_info_section.get("customers_say", "N/A")
                
                amazon_products.append({
                    "query": product_name,
                    "platform": "Amazon",
                    "source": "amazon",
                    "title": product_results.get("title", "N/A"),
                    "description": product_results.get("description", "N/A"),
                    "star_ratings": reviews_summary.get("rating", "N/A"),
                    "total_reviews": reviews_summary.get("total_reviews", "N/A"),
                    "customers_say_summary": customers_say,
                    "asin": asin,
                    "link": product.get("link", "N/A"),
                    "price": product_results.get("price", "N/A"),
                    "availability": product_results.get("availability", "N/A")
                })
                
                print(f"    ✓ Scraped Amazon product: {product_results.get('title', 'Unknown')[:50]}...")
                
            except Exception as e:
                print(f"    [Warning] Error scraping ASIN {asin}: {e}")
                continue
        
        return amazon_products
        
    except Exception as e:
        print(f"[Error] Error scraping Amazon for '{product_name}': {e}")
        return []

# --- Flipkart Scraper ---

def scrape_flipkart_product(product_name):
    serpapi_key = get_serpapi_key()
    if not serpapi_key:
        print("[Warning] SERP_API_KEY not found for Flipkart scraping.")
        return []
        
    print(f"[+] Scraping Flipkart for '{product_name}'...")
    try:
        search_params = {"engine": "flipkart", "q": product_name, "api_key": serpapi_key}
        search = GoogleSearch(search_params)
        search_results = search.get_dict()
        
        organic_results = search_results.get("organic_results", [])
        if not organic_results:
            print(f"    [Warning] No Flipkart results found for '{product_name}'")
            return []
        
        first_product = organic_results[0]
        pid = first_product.get("product_id")
        
        if not pid:
            print(f"    [Warning] No product ID found for '{product_name}'")
            return []
        
        product_params = {"engine": "flipkart_product", "product_id": pid, "api_key": serpapi_key}
        product_search = GoogleSearch(product_params)
        product_info = product_search.get_dict()
        
        return [{
            "query": product_name,
            "platform": "Flipkart",
            "source": "flipkart",
            "title": product_info.get("product_name", "N/A"),
            "description": product_info.get("description", "N/A"),
            "star_ratings": product_info.get("ratings", "N/A"),
            "total_reviews": product_info.get("total_reviews", "N/A"),
            "product_id": pid,
            "link": first_product.get("link", "N/A")
        }]
        
    except Exception as e:
        print(f"[Error] Error scraping Flipkart for '{product_name}': {e}")
        return []

# --- Croma Scraper (From first file) ---

def scrape_croma_site(keyword, brands, product_types):
    """Croma site scraper with correct filtering."""
    serpapi_key = get_serpapi_key()
    if not serpapi_key:
        print(f"[Warning] SERP_API_KEY not found for Croma scraping.")
        return []
        
    print(f"[+] Scraping Croma for '{keyword}'...")
    try:
        params = {"engine": "google", "q": f"site:croma.com {keyword} review", "api_key": serpapi_key}
        search = GoogleSearch(params)
        results = search.get_dict()
        
        data = []
        organic_results = results.get("organic_results", [])
        
        for res in organic_results:
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            
            if post_matches_filters(title, snippet, brands, product_types):
                data.append({
                    "query": keyword,
                    "platform": "Croma",
                    "source": "croma.com",
                    "title": title,
                    "link": res.get("link"),
                    "snippet": snippet
                })
        
        print(f"    [✓] Found {len(data)} relevant results from Croma")
        return data
        
    except Exception as e:
        print(f"[Error] Error scraping Croma for '{keyword}': {e}")
        return []

# --- Reliance Digital Enhanced Scraper ---

def scrape_reliance_digital_enhanced(keyword, brands, product_types):
    """Enhanced Reliance Digital scraper to get star ratings."""
    serpapi_key = get_serpapi_key()
    if not serpapi_key:
        print(f"[Warning] SERP_API_KEY not found for Reliance Digital scraping.")
        return []
        
    print(f"[+] Scraping Reliance Digital for '{keyword}'...")
    try:
        params = {
            "engine": "google", 
            "q": f"site:reliancedigital.in {keyword}", 
            "api_key": serpapi_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        data = []
        organic_results = results.get("organic_results", [])
        
        for res in organic_results:
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            link = res.get("link", "")
            
            if post_matches_filters(title, snippet, brands, product_types):
                # Try to extract rating from snippet if available
                star_rating = "N/A"
                
                # Look for rating patterns in snippet
                rating_patterns = [
                    r'(\d+\.?\d*)\s*/\s*5',  # "4.2/5" or "4 / 5"
                    r'(\d+\.?\d*)\s*out of 5',  # "4.2 out of 5"
                    r'rating[:\s]*(\d+\.?\d*)',  # "rating: 4.2"
                    r'(\d+\.?\d*)\s*stars?',  # "4.2 stars"
                    r'(\d+\.?\d*)\s*star rating',  # "4.2 star rating"
                ]
                
                for pattern in rating_patterns:
                    match = re.search(pattern, snippet.lower())
                    if match:
                        star_rating = f"{match.group(1)}/5"
                        break
                
                # If no rating in snippet, try to fetch more specific rating info
                if star_rating == "N/A":
                    try:
                        # Use Google search to find rating information specifically
                        rating_search_params = {
                            "engine": "google",
                            "q": f'"{title}" site:reliancedigital.in rating review',
                            "api_key": serpapi_key
                        }
                        rating_search = GoogleSearch(rating_search_params)
                        rating_results = rating_search.get_dict()
                        
                        # Check if rating info is in the search results
                        for rating_result in rating_results.get("organic_results", [])[:3]:
                            rating_snippet = rating_result.get("snippet", "")
                            for pattern in rating_patterns:
                                match = re.search(pattern, rating_snippet.lower())
                                if match:
                                    star_rating = f"{match.group(1)}/5"
                                    break
                            if star_rating != "N/A":
                                break
                                
                    except Exception as e:
                        print(f"    [Warning] Could not fetch detailed rating for {title[:30]}...")
                
                data.append({
                    "query": keyword,
                    "platform": "Reliancedigital",
                    "source": "reliancedigital.in",
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "star_ratings": star_rating
                })
        
        print(f"    [✓] Found {len(data)} relevant results from Reliance Digital")
        return data
        
    except Exception as e:
        print(f"[Error] Error scraping Reliance Digital for '{keyword}': {e}")
        return []

# --- Main Execution Function ---

def run_all_scrapers(brands, product_types, reddit_subreddits):
    """
    Runs all e-commerce and Reddit scrapers for a list of brands and product types.
    """
    all_data = []
    
    # Create search queries for e-commerce platforms
    search_queries = [f"{brand} {ptype}" for brand, ptype in product(brands, product_types)]
    
    print(f"[+] Created {len(search_queries)} search queries")
    
    # Use ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        
        # Schedule e-commerce scraping tasks (Amazon, Flipkart, Croma, and Reliance Digital)
        for query in search_queries[:12]:  # Limit queries to avoid too many API calls
            futures.append(executor.submit(scrape_amazon_product, query))
            futures.append(executor.submit(scrape_flipkart_product, query))
            futures.append(executor.submit(scrape_croma_site, query, brands, product_types))
            futures.append(executor.submit(scrape_reliance_digital_enhanced, query, brands, product_types))
            
        # Schedule Reddit scraping
        futures.append(executor.submit(scrape_reddit, reddit_subreddits, brands, product_types, query_mode="broad"))
        futures.append(executor.submit(scrape_reddit, reddit_subreddits, brands, product_types, query_mode="targeted"))
        
        # Collect all results
        for future in as_completed(futures):
            try:
                result = future.result()
                if isinstance(result, list):
                    all_data.extend(result)
            except Exception as e:
                print(f"[Error] Task failed: {e}")
    
    return all_data

if __name__ == "__main__":
    brands = ["atomberg", "usha", "havells", "crompton", "orient", "bajaj", "luminous", "lg", "vguard", "honeywell", "polycab"]
    product_types = ["smart fan", "bldc fan", "fan"]
    reddit_subreddits = ["smarthome", "homeautomation", "india", "mumbai", "Chennai", "Frugal_Ind", "Kerala", "delhi", "kolkata"]
    
    print("[+] Starting data scraping process...")
    print(f"[+] Targeting brands: {', '.join(brands)}")
    print(f"[+] Targeting products: {', '.join(product_types)}")
    print(f"[+] Searching subreddits: {', '.join(reddit_subreddits)}")
    print("-" * 60)
    
    # Check API keys
    if not get_serpapi_key():
        print("[Warning] SERP_API_KEY not found. E-commerce scraping will be skipped.")
    
    if not get_reddit_client():
        print("[Warning] Reddit credentials not found. Reddit scraping will be skipped.")
    
    try:
        data = run_all_scrapers(brands, product_types, reddit_subreddits)
        
        if data:
            saved_file = save_data_to_json(data, "feedback_smart_fan")
            if saved_file:
                print(f"\n[✓] Scraping completed successfully!")
                print(f"[✓] File saved: {os.path.basename(saved_file)}")
            else:
                print("\n[✗] Scraping completed but failed to save data")
        else:
            print("\n[!] No data scraped. Please check your API keys and network connection.")
            
        # Print summary
        print(f"\n=== SCRAPING SUMMARY ===")
        platforms = {}
        for item in data:
            platform = item.get('platform', item.get('source', 'unknown'))
            platforms[platform] = platforms.get(platform, 0) + 1
        
        for platform, count in platforms.items():
            print(f"{platform.capitalize()}: {count} items")
        print(f"Total items: {len(data)}")
        
        if data and len(data) > 0:
            print(f"\n=== SAMPLE RESULTS (First 2) ===")
            for i, item in enumerate(data[:2]):
                print(f"\nItem {i+1}:")
                print(f"  Platform: {item.get('platform', item.get('source', 'N/A'))}")
                print(f"  Title: {item.get('title', 'N/A')[:100]}...")
                if item.get('query'):
                    print(f"  Query: {item.get('query')}")
        print("=" * 50)
        
    except KeyboardInterrupt:
        print("\n[!] Scraping interrupted by user.")
    except Exception as e:
        print(f"\n[Error] Unexpected error: {e}")