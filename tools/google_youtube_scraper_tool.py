import os
import json
import time
import requests
from datetime import datetime
from config import serp_api
from deep_translator import GoogleTranslator  # pip install deep-translator
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs

# ====== CONFIG ======
SERP_API_KEY = os.getenv("SERP_API_KEY") or serp_api.get("api_key")
BASE_URL = "https://serpapi.com/search"
SERP_BASE_URL = "https://serpapi.com/search"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def extract_video_id(youtube_url):
    """
    Extract the YouTube video ID from either a full YouTube URL or a youtu.be short link.
    """
    if not youtube_url:
        return None

    parsed_url = urlparse(youtube_url)

    # Standard watch URL
    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]

    # Shortened youtu.be URL
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None


def fetch_serp_results(keyword, engine="google", num_results=10):
    params = {
        "api_key": SERP_API_KEY,
        "engine": engine,
        "num": num_results
    }
    if engine == "google":
        params["q"] = keyword
    elif engine == "youtube":
        params["search_query"] = keyword
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"[!] Attempt {attempt+1} failed for {engine} '{keyword}': {e}")
            time.sleep(RETRY_DELAY * (2 ** attempt))

    raise RuntimeError(f"[✗] Failed to fetch results for '{keyword}' after {MAX_RETRIES} attempts")


def parse_google_results(json_data):
    results = []
    for r in json_data.get("organic_results", []):
        link = r.get("link", "")
        snippet = r.get("snippet", "")

        if "google.com" in link and "reviews" in link:
            reviews = fetch_google_reviews(link)
            results.append({
                "title": r.get("title", ""),
                "link": link,
                "snippet": snippet,
                "platform": "Google Reviews",
                "reviews": reviews
            })
        else:
            results.append({
                "title": r.get("title", ""),
                "link": link,
                "snippet": snippet,
                "platform": "Google"
            })
    return results


def parse_youtube_results(json_data):
    results = []
    for r in json_data.get("video_results", []):
        link = r.get("link", "")
        video_id = r.get("video_id") or extract_video_id(link)

        comments = fetch_youtube_comments(video_id)
        # Corrected line: Pass the full URL to the function
        transcript = fetch_youtube_transcript(link)

        results.append({
            "title": r.get("title", ""),
            "link": link,
            "snippet": r.get("description", ""),
            "platform": "YouTube",
            "comments": comments,
            "transcript": transcript
        })
    return results


def fetch_google_reviews(place_url):
    params = {
        "api_key": SERP_API_KEY,
        "engine": "google_maps_reviews",
        "hl": "en",
        "place_id": extract_place_id(place_url)
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [rev.get("snippet") for rev in data.get("reviews", []) if rev.get("snippet")]
    except Exception as e:
        print(f"[!] Failed to fetch Google reviews: {e}")
        return []


def extract_place_id(url):
    if "placeid=" in url:
        return url.split("placeid=")[-1].split("&")[0]
    return url


def detect_and_translate(text, target_lang):
    """
    Detects the language of the text and translates it to the target language if needed.
    """
    try:
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
        return translated_text
    except Exception:
        # If translation fails, return the original text
        return text


def fetch_youtube_comments(video_id, max_comments=50):
    """
    Fetches YouTube comments using SerpAPI and translates them to English.
    """
    print(f"[+] Extracting comments using SerpAPI for video ID: {video_id}")
    
    if not video_id or not SERP_API_KEY:
        print("[!] Missing video_id or SERP_API_KEY")
        return {'success': False, 'error': 'Missing video_id or API key', 'comments': []}
    
    params = {
        "api_key": SERP_API_KEY,
        "engine": "youtube_comments",
        "video_id": video_id,
        "num": min(max_comments, 100)  # SerpAPI limit
    }
    
    try:
        print(f"[+] Making SerpAPI request for video: {video_id}")
        resp = requests.get(SERP_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        comments = []
        raw_comments = data.get("comments", [])
        
        if not raw_comments:
            print(f"[!] No comments found in SerpAPI response")
            return {'success': True, 'comments': [], 'total_fetched': 0}
        
        print(f"[+] Processing {len(raw_comments)} comments from SerpAPI")
        
        for comment_data in raw_comments[:max_comments]:
            if not comment_data.get("text"):
                continue
            
            original_text = comment_data.get("text", "")
            
            # Clean up the text (remove extra whitespace, etc.)
            original_text = ' '.join(original_text.split())
            
            # Translate if not in English
            translated_text = detect_and_translate(original_text, 'en')
            
            comment_entry = {
                'text': translated_text,
                'original_text': original_text if original_text != translated_text else None,
                'author': comment_data.get("author", "Unknown"),
                'likes': comment_data.get("likes", 0),
                'time': comment_data.get("time", ""),
                'replies_count': comment_data.get("replies", 0),
                'source': 'serpapi'
            }
            
            comments.append(comment_entry)
        
        print(f"[✓] Successfully processed {len(comments)} comments using SerpAPI")
        
        return {
            'success': True,
            'comments': comments,
            'total_fetched': len(comments),
            'source': 'serpapi'
        }
    
    except requests.exceptions.RequestException as e:
        print(f"[!] SerpAPI request failed for {video_id}: {e}")
        return {'success': False, 'error': f'Request failed: {e}', 'comments': []}
    except Exception as e:
        print(f"[!] Failed to fetch YouTube comments via SerpAPI for {video_id}: {e}")
        return {'success': False, 'error': str(e), 'comments': []}


def fetch_youtube_transcript(youtube_url):
    """
    Fetches the transcript for a given YouTube video URL using the same method
    as your test_fetch_transcript() code.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        print("\n❌ Could not extract video ID from the provided URL.")
        return "Invalid YouTube URL."

    print(f"Attempting to fetch transcript for video ID: {video_id}")
    try:
        # Create API instance
        ytt_api = YouTubeTranscriptApi()

        # Try fetching the transcript with preferred languages
        fetched_transcript = ytt_api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])

        # Extract text from all snippets
        transcript_text = " ".join([snippet.text for snippet in fetched_transcript])

        if transcript_text:
            print("\n✅ Transcript fetched successfully!")
            print("-" * 50)
            print(f"Video ID: {fetched_transcript.video_id}")
            print(f"Language: {fetched_transcript.language} ({fetched_transcript.language_code})")
            print(f"Auto-generated: {fetched_transcript.is_generated}")
            print(f"Number of snippets: {len(fetched_transcript)}")
            print(f"Total length: {len(transcript_text)} characters")
            print("\nPreview:")
            print(transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text)
            print("-" * 50)
            return transcript_text
        else:
            print("\n❌ Transcript was empty.")
            return "Transcript empty."

    except NoTranscriptFound:
        print("\n❌ No transcript found for this video.")
        print("Let me check what transcripts are available...")
        list_available_transcripts(video_id)
        return "No transcript found."
    except TranscriptsDisabled:
        print("\n❌ Transcripts are disabled for this video.")
        return "Transcripts disabled."
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        return f"Error: {e}"


def list_available_transcripts(video_id):
    """
    Lists all available transcripts for a given YouTube video ID.
    """
    print(f"Listing available transcripts for video ID: {video_id}")
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        if transcript_list:
            print("\n✅ Available transcripts:")
            transcripts = []
            for transcript in transcript_list:
                print(f"  - Language: {transcript.language}")
                print(f"    Code: {transcript.language_code}")
                print(f"    Auto-generated: {transcript.is_generated}")
                print(f"    Translatable: {transcript.is_translatable}")
                if transcript.translation_languages:
                    print(f"    Can translate to: {len(transcript.translation_languages)} languages")
                print()
                transcripts.append(transcript)
            return transcripts
        else:
            print("\n❌ No transcripts available.")
            return []

    except Exception as e:
        print(f"\n❌ Error listing transcripts: {e}")
        print(f"Error type: {type(e).__name__}")
        return []


def fetch_any_available_transcript(video_id):
    """
    Fetches the first available transcript for a video.
    """
    print(f"Attempting to fetch any available transcript for video ID: {video_id}")
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        if not transcript_list:
            print("\n❌ No transcripts available.")
            return "No transcripts available."

        print(f"\nFound {len(transcript_list)} available transcript(s)")
        first_transcript = transcript_list[0]
        print(f"Fetching: {first_transcript.language} ({first_transcript.language_code})")

        fetched_transcript = first_transcript.fetch()
        transcript_text = " ".join([snippet.text for snippet in fetched_transcript])

        if transcript_text:
            print("\n✅ Transcript fetched successfully!")
            print("-" * 50)
            print(f"Video ID: {fetched_transcript.video_id}")
            print(f"Language: {fetched_transcript.language} ({fetched_transcript.language_code})")
            print(f"Auto-generated: {fetched_transcript.is_generated}")
            print(f"Number of snippets: {len(fetched_transcript)}")
            print(f"Total length: {len(transcript_text)} characters")
            print("\nPreview:")
            print(transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text)
            print("-" * 50)
            return transcript_text
        else:
            print("\n❌ Transcript was empty.")
            return "Transcript was empty."

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        return f"Error: {e}"


def fetch_transcript_with_fallback(video_id):
    """
    Tries multiple approaches to fetch a transcript.
    """
    print(f"Trying multiple methods to fetch transcript for video ID: {video_id}")
    ytt_api = YouTubeTranscriptApi()

    # Method 1: Default fetch
    try:
        print("\n1. Trying default fetch (no language specified)...")
        fetched_transcript = ytt_api.fetch(video_id)
        transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
        if transcript_text:
            print("✅ Success with default fetch!")
            return transcript_text
    except Exception as e:
        print(f"   Failed: {e}")

    # Method 2: Any available transcript
    return fetch_any_available_transcript(video_id)


def scrape_keywords(keywords, num_results=10, save=True):
    all_results = []
    for kw in keywords:
        print(f"[+] Scraping Google for '{kw}'...")
        google_data = fetch_serp_results(kw, engine="google", num_results=num_results)
        all_results.extend(parse_google_results(google_data))

        print(f"[+] Scraping YouTube for '{kw}'...")
        youtube_data = fetch_serp_results(kw, engine="youtube", num_results=num_results)
        all_results.extend(parse_youtube_results(youtube_data))

    if save:
        os.makedirs("data/raw", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/raw/google_youtube_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[✓] Saved results to {filename}")

    return all_results


if __name__ == "__main__":
    # Define a list of keywords to scrape.
    test_keywords = ["smart fan", "smartfan", "bldc fan", "smart ceiling fan", "atomberg", "usha", "havells", "crompton", "orient", "bajaj", "luminous", "lg", "vguard", "honeywell", "polycab"]
    
    # Scrape the data for the keywords.
    # The `save=False` argument prevents it from saving to a file in this example.
    all_scraped_results = scrape_keywords(test_keywords, num_results=5, save=False)
    
    # Print the results in a human-readable JSON format.
    print(json.dumps(all_scraped_results, indent=2, ensure_ascii=False))

