# insight_pipeline.py
import os
import json
import torch
import numpy as np
import nltk
import time
import glob
import re
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
from groq import Groq
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# IMPORTANT: Ensure the NLTK VADER lexicon and punkt_tab are downloaded.
try:
    print("[+] Downloading NLTK resources...")
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)   # Added punkt_tab for newer NLTK versions
    print("[✓] NLTK resources downloaded successfully.")
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

# --- Sentiment Analyzer Agent ---

# Load transformer model
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading BERT model: {e}. Falling back to VADER.")
    tokenizer = None
    model = None

# Initialize VADER
vader = SentimentIntensityAnalyzer()

def bert_sentiment(text):
    """Get sentiment prediction and confidence from BERT."""
    if not tokenizer or not model:
        return None, 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class_id = int(np.argmax(probs))

    label = model.config.id2label[predicted_class_id].lower()
    confidence = float(probs[predicted_class_id])
    return label, confidence


def vader_sentiment(text):
    """Get sentiment prediction and score from VADER."""
    scores = vader.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, compound


def combined_sentiment(text):
    """Combine BERT + VADER for better accuracy."""
    if not text or len(text.strip()) < 3:
        return "neutral", 0.0

    bert_label, bert_conf = bert_sentiment(text)
    vader_label, vader_score = vader_sentiment(text)

    # Ensemble rule
    if bert_label and bert_label == vader_label:
        return bert_label, round((bert_conf + abs(vader_score)) / 2, 3)
    elif bert_label:
        if bert_conf > abs(vader_score):
            return bert_label, bert_conf
        else:
            return vader_label, abs(vader_score)
    else:
        # Fallback to VADER if BERT fails to load
        return vader_label, abs(vader_score)

def get_most_sentimental_sentence(text, target_sentiment):
    """
    Finds the sentence in a text with the highest confidence for a given sentiment.
    """
    if not text or len(text.strip()) < 3:
        return ""
    
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError as e:
        # Fallback to simple sentence splitting if NLTK fails
        print(f"NLTK tokenization failed: {e}. Using simple sentence splitting.")
        sentences = text.split('. ')
    
    best_sentence = ""
    best_confidence = 0.0
    for sentence in sentences:
        if sentence.strip():  # Only process non-empty sentences
            label, conf = combined_sentiment(sentence)
            if label == target_sentiment and conf > best_confidence:
                best_confidence = conf
                best_sentence = sentence
    return best_sentence.strip()


def extract_mentions():
    """
    Loads and processes scraped data from the most recent JSON files in the `data/raw` directory,
    including new files from X.
    """
    print("[+] Loading scraped data from JSON files...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "raw")
    
    mentions = []

    # Find the most recent YouTube data file
    youtube_files = glob.glob(os.path.join(data_dir, "google_youtube_*.json"))
    youtube_file = max(youtube_files, key=os.path.getctime) if youtube_files else None

    # Find the most recent feedback data file
    feedback_files = glob.glob(os.path.join(data_dir, "feedback_smart_fan_*.json"))
    feedback_file = max(feedback_files, key=os.path.getctime) if feedback_files else None
    
    # Find all X data files and load them
    x_files = glob.glob(os.path.join(data_dir, "x_fan_reviews_*.json"))
    x_data = []
    if x_files:
        print("[✓] Found and loading X files...")
        for x_file in sorted(x_files, key=os.path.getctime, reverse=True):
            try:
                with open(x_file, 'r', encoding='utf-8') as f:
                    x_data.extend(json.load(f))
            except Exception as e:
                print(f"Error loading X data from {os.path.basename(x_file)}: {e}")
    else:
        print("[x] No X files found in the data/raw directory.")

    # Load YouTube data
    if youtube_file and os.path.exists(youtube_file):
        print(f"[✓] Found and loading YouTube file: {os.path.basename(youtube_file)}")
        try:
            with open(youtube_file, 'r', encoding='utf-8') as f:
                mentions.extend(json.load(f))
        except Exception as e:
            print(f"Error loading YouTube data: {e}")
    else:
        print("[x] No YouTube file found in the data/raw directory.")

    # Load general feedback data
    if feedback_file and os.path.exists(feedback_file):
        print(f"[✓] Found and loading feedback file: {os.path.basename(feedback_file)}")
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                mentions.extend(json.load(f))
        except Exception as e:
            print(f"Error loading feedback data: {e}")
    else:
        print("[x] No feedback file found in the data/raw directory.")
        
    mentions.extend(x_data)
    
    return mentions

def run_sentiment_analysis():
    """
    Runs sentiment analysis on all mentions by explicitly handling the different
    data structures from the various scraping sources.
    """
    mentions = extract_mentions()
    for mention in mentions:
        texts = []
        platform = mention.get("platform", "").lower()

        if platform == "reddit":
            if mention.get("title"):
                texts.append(mention["title"])
            if isinstance(mention.get("comments"), list):
                texts.extend([c for c in mention["comments"] if isinstance(c, str)])
        elif platform == "youtube":
            if mention.get("title"):
                texts.append(mention["title"])
            if mention.get("snippet"):
                texts.append(mention["snippet"])
            if mention.get("transcript"):
                texts.append(mention["transcript"])
            # Handle the comments object from the scraper, but ignore if empty
            if isinstance(mention.get("comments"), dict) and mention["comments"].get("comments"):
                texts.extend([c for c in mention["comments"]["comments"] if isinstance(c, str)])
        elif platform == "x":
            # X data has a "content" field for the main post and a list of "replies"
            if mention.get("content"):
                texts.append(mention["content"])
            if isinstance(mention.get("replies"), list):
                texts.extend([r["content"] for r in mention["replies"] if isinstance(r, dict) and "content" in r])
        else: # Assumes e-commerce platforms like Google, Amazon, Flipkart, etc.
            if mention.get("title"):
                texts.append(mention["title"])
            if mention.get("snippet"):
                texts.append(mention["snippet"])
            if isinstance(mention.get("reviews"), list):
                texts.extend([r for r in mention["reviews"] if isinstance(r, str)])
        
        # Filter out any non-text values before analysis
        texts_to_analyze = [t for t in texts if t and isinstance(t, str)]

        if not texts_to_analyze:
            mention["sentiment"] = "neutral"
            mention["sentiment_confidence"] = 0.0
            continue

        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        confidences = []

        for text in texts_to_analyze:
            label, conf = combined_sentiment(text)
            sentiment_counts[label] += 1
            confidences.append(conf)

        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        dominant_confidences = [conf for i, conf in enumerate(confidences)
                                 if list(sentiment_counts.keys())[list(sentiment_counts.values()).index(max(sentiment_counts.values()))] == dominant_sentiment]
        avg_conf = round(sum(dominant_confidences) / len(dominant_confidences), 3) if dominant_confidences else 0.0

        mention["sentiment"] = dominant_sentiment
        mention["sentiment_confidence"] = avg_conf

    print(f"[✓] Sentiment analysis completed for {len(mentions)} mentions.")
    return mentions


# --- Insight Generator Agent ---

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

BRANDS = [
    "atomberg", "orient", "havells", "lg", "crompton", "polycab", "bajaj", "usha", "luminous", "vguard"
]

POSITIVE_KEYWORDS = ["best", "effective", "good", "happy", "nice", "love", "excellent", "great", "saving", "power", "silent", "quiet"]
NEGATIVE_KEYWORDS = ["bad", "worst", "hate", "broke", "no", "problem", "issue", "service", "customer", "disappointed", "expensive", "cost", "noise", "return", "refund"]

def parse_star_rating(rating_str):
    """Parses a string like '4.5/5' and returns the float value, or None if invalid."""
    if not isinstance(rating_str, str) or '/' not in rating_str:
        return None
    try:
        score, total = rating_str.split('/')
        return float(score.strip())
    except (ValueError, IndexError):
        return None


def generate_insights(use_groq=False):
    """Generates an insights report based on analyzed data."""
    mentions = run_sentiment_analysis()

    if not mentions:
        print("No mentions found.")
        return

    brand_mentions = defaultdict(int)
    brand_positive_mentions = defaultdict(int)
    brand_engagement = defaultdict(int)
    
    atomberg_positive_review_snippets = []
    atomberg_negative_review_snippets = []
    competitor_positive_review_snippets = defaultdict(list)

    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    sentiment_keyword_counts = defaultdict(Counter)
    
    atomberg_transcript_sentiment = {"positive": 0, "neutral": 0, "negative": 0}

    # New metrics
    reliance_croma_mentions = defaultdict(int)
    total_reliance_croma_mentions = 0
    brand_star_ratings = defaultdict(list)
    high_rated_features = []
    
    # --- BUG FIX: Initialize the sentiment_reasons dictionary ---
    sentiment_reasons = defaultdict(list)

    for mention in mentions:
        # Use the content or title for analysis
        text_content = (mention.get("content") or mention.get("title") or mention.get("snippet") or "").lower()
        sentiment = mention.get("sentiment", "neutral")
        sentiments[sentiment] += 1

        engagement_score = 0
        if mention.get("platform") == "Reddit":
            engagement_score = mention.get("num_comments", 0)
        elif mention.get("platform") == "YouTube":
            if mention.get("transcript"):
                t_label, _ = combined_sentiment(mention["transcript"])
                atomberg_transcript_sentiment[t_label] += 1
            engagement_score = 0
        elif mention.get("platform") in ["amazon", "flipkart", "reliance digital", "croma"]:
            engagement_score = len(mention.get("reviews", []))
        elif mention.get("platform") == "X":
            engagement_score = mention.get("replyCount", 0) + mention.get("retweetCount", 0) + mention.get("likeCount", 0)

        mentioned_brands = [brand.lower() for brand in BRANDS if brand.lower() in text_content]
        
        if mentioned_brands:
            for brand in mentioned_brands:
                brand_mentions[brand] += 1
                if sentiment == "positive":
                    brand_positive_mentions[brand] += 1
                brand_engagement[brand] += engagement_score
            
            # --- Collect sample reviews and competitor positives ---
            if "atomberg" in mentioned_brands:
                if sentiment == "positive" and len(atomberg_positive_review_snippets) < 2:
                    snippet = get_most_sentimental_sentence(text_content, "positive")
                    if snippet:
                        atomberg_positive_review_snippets.append(snippet)
                elif sentiment == "negative" and len(atomberg_negative_review_snippets) < 2:
                    snippet = get_most_sentimental_sentence(text_content, "negative")
                    if snippet:
                        atomberg_negative_review_snippets.append(snippet)
            
            for brand in mentioned_brands:
                if brand != "atomberg" and sentiment == "positive" and len(competitor_positive_review_snippets[brand]) < 2:
                    snippet = get_most_sentimental_sentence(text_content, "positive")
                    if snippet:
                        competitor_positive_review_snippets[brand].append(snippet)
            
            # Use curated keywords for counting
            for word in text_content.split():
                clean_word = word.strip(".,!?()[]\"'").lower()
                if clean_word in POSITIVE_KEYWORDS:
                    sentiment_keyword_counts["positive"][clean_word] += 1
                elif clean_word in NEGATIVE_KEYWORDS:
                    sentiment_keyword_counts["negative"][clean_word] += 1
        
        # --- BUG FIX: Populate sentiment_reasons dictionary with sentences ---
        if sentiment == "positive":
            sentence = get_most_sentimental_sentence(text_content, "positive")
            if sentence:
                sentiment_reasons['positive'].append(sentence)
        elif sentiment == "negative":
            sentence = get_most_sentimental_sentence(text_content, "negative")
            if sentence:
                sentiment_reasons['negative'].append(sentence)
                
        # --- New metrics calculation ---
        if mention.get("platform") in ["Reliancedigital", "Croma"]:
            total_reliance_croma_mentions += 1
            for brand in mentioned_brands:
                reliance_croma_mentions[brand] += 1
        
        if mention.get("platform") == "Reliancedigital":
            star_rating = parse_star_rating(mention.get("star_ratings"))
            if star_rating is not None:
                for brand in mentioned_brands:
                    brand_star_ratings[brand].append(star_rating)
                
                # Check for high-rated products
                if star_rating > 4:
                    high_rated_features.append({
                        "query": mention.get("query"),
                        "title": mention.get("title"),
                        "snippet": mention.get("snippet")
                    })

    total_brand_mentions = sum(brand_mentions.values())
    total_engagement = sum(brand_engagement.values())
    total_positive_mentions = sum(brand_positive_mentions.values())

    MS = (brand_mentions["atomberg"] / total_brand_mentions) if total_brand_mentions else 0
    ES = (brand_engagement["atomberg"] / total_engagement) if total_engagement else 0
    SoPV = (brand_positive_mentions["atomberg"] / total_positive_mentions) if total_positive_mentions else 0

    SoV_score = 0.5 * MS + 0.3 * ES + 0.2 * SoPV

    print("\n=== Atomberg Share of Voice Metrics ===")
    print(f"Mentions Share (MS): {MS:.2%} ({brand_mentions['atomberg']} / {total_brand_mentions})")
    print(f"Engagement Share (ES): {ES:.2%} ({brand_engagement['atomberg']} / {total_engagement})")
    print(f"Share of Positive Voice (SoPV): {SoPV:.2%} ({brand_positive_mentions['atomberg']} / {total_positive_mentions})")
    print(f"Weighted SoV Score: {SoV_score:.2%}")

    print("\n=== Brand Performance Overview ===")
    print(f"{'Brand':<15} | {'Mentions':<10} | {'Positive':<10} | {'Engagement':<10} | {'Weighted SoV':<15} | {'Accessibility':<15} | {'Avg. Star Rating':<15}")
    print("-" * 115)
    sorted_brands = sorted(brand_mentions.keys(), key=lambda b: brand_mentions[b], reverse=True)
    for brand in sorted_brands:
        if brand_mentions[brand] > 0:
            pos_mentions = brand_positive_mentions[brand]
            engagement = brand_engagement[brand]
            mentions_count = brand_mentions[brand]
            
            brand_MS = (brand_mentions[brand] / total_brand_mentions) if total_brand_mentions else 0
            brand_ES = (brand_engagement[brand] / total_engagement) if total_engagement else 0
            brand_SoPV = (brand_positive_mentions[brand] / total_positive_mentions) if total_positive_mentions else 0
            brand_SoV_score = 0.5 * brand_MS + 0.3 * brand_ES + 0.2 * brand_SoPV

            accessibility_score = (reliance_croma_mentions[brand] / total_reliance_croma_mentions) if total_reliance_croma_mentions else 0
            avg_star_rating = sum(brand_star_ratings[brand]) / len(brand_star_ratings[brand]) if brand_star_ratings[brand] else 'N/A'
            
            print(f"{brand.capitalize():<15} | {mentions_count:<10} | {pos_mentions:<10} | {engagement:<10} | {brand_SoV_score:.2%} | {accessibility_score:.2%} | {avg_star_rating}")

    print("\n=== Sample Atomberg Reviews (Positive & Negative Snippets) ===")
    print("--- Positive ---")
    for i, review in enumerate(atomberg_positive_review_snippets):
        print(f"{i+1}. {review}")
    print("\n--- Negative ---")
    for i, review in enumerate(atomberg_negative_review_snippets):
        print(f"{i+1}. {review}")

    print("\n=== Best features that made the product get above 4/5 star rating in Reliance Digital ===")
    if high_rated_features:
        for i, feature in enumerate(high_rated_features[:5]):
            print(f"{i+1}. {feature['query']} -> {feature['title']} -> {feature['snippet']}")
    else:
        print("No high-rated features found in Reliance Digital data.")

    print("\n=== Atomberg YouTube Transcript Sentiment Breakdown ===")
    print("Transcript Sentiment:", atomberg_transcript_sentiment)

    top_competitors = [b for b in sorted_brands if b != "atomberg" and brand_mentions[b] > 0][:2]

    competitor_plus_points = defaultdict(list)
    for mention in mentions:
        text_content = (mention.get("content") or mention.get("title") or mention.get("snippet") or "").lower()
        sentiment = mention.get("sentiment", "neutral")
        
        if sentiment == "positive":
            for brand in top_competitors:
                if brand in text_content:
                    positive_indicators = []
                    if any(word in text_content for word in ["service", "support", "customer"]):
                        positive_indicators.append("Good Customer Service")
                    if any(word in text_content for word in ["cheap", "affordable", "price", "cost"]):
                        positive_indicators.append("Affordable Pricing")
                    if any(word in text_content for word in ["durable", "lasting", "quality", "build"]):
                        positive_indicators.append("Build Quality & Durability")
                    if any(word in text_content for word in ["speed", "wind", "air", "flow"]):
                        positive_indicators.append("High Wind Speed")
                    if any(word in text_content for word in ["silent", "quiet", "noise"]):
                        positive_indicators.append("Silent Operation")
                    if any(word in text_content for word in ["energy", "power", "saving", "efficient"]):
                        positive_indicators.append("Energy Efficiency")
                    if any(word in text_content for word in ["design", "look", "appearance", "style"]):
                        positive_indicators.append("Attractive Design")
                    if any(word in text_content for word in ["remote", "control", "smart", "app"]):
                        positive_indicators.append("Smart Features")
                    
                    competitor_plus_points[brand].extend(positive_indicators)

    print(f"\n=== Plus Points of Atomberg's Competitors ===")
    for brand in top_competitors[:2]:
        if competitor_plus_points[brand]:
            unique_points = list(set(competitor_plus_points[brand]))
            points_with_count = [(point, competitor_plus_points[brand].count(point)) for point in unique_points]
            points_with_count.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n**{brand.capitalize()}:**")
            for point, count in points_with_count[:3]:
                print(f"  • {point} (mentioned {count} times)")
        else:
            print(f"\n**{brand.capitalize()}:** No specific positive attributes identified from the data.")
    print("\n=== Top Keywords by Sentiment ===")
    print("Top 5 Positive Keywords:", sentiment_keyword_counts["positive"].most_common(5))
    print("Top 5 Negative Keywords:", sentiment_keyword_counts["negative"].most_common(5))
    print("Top 5 Neutral Keywords: (Filtered for irrelevant words)")


    competitor_insights = ""
    for brand in top_competitors:
        positive_reviews = "\n- ".join(competitor_positive_review_snippets[brand])
        competitor_insights += f"\n### {brand.capitalize()} Positive Customer Feedback\n- {positive_reviews}\n"
    prompt = f"""
    You are a brand marketing strategist and business economist analyzing the competitive landscape for BLDC and smart ceiling fans in India. Your goal is to provide actionable insights that increase Atomberg's Share of Voice (SoV), profitability, and customer satisfaction simultaneously.

    Here are the key metrics for Atomberg vs. its competition, based on online mentions and engagement:

    ### Overall Atomberg Performance
    - **Mentions Share (MS):** {MS:.2%}
    - **Engagement Share (ES):** {ES:.2%}
    - **Share of Positive Voice (SoPV):** {SoPV:.2%}
    - **Weighted SoV Score:** {SoV_score:.2%}

    ### Competitive Brand Data
    {json.dumps({b: {"mentions": brand_mentions[b], "positive_mentions": brand_positive_mentions[b], "engagement": brand_engagement[b]} for b in sorted_brands if brand_mentions[b] > 0}, indent=2)}

    ### Competitor Strengths Analysis
    Based on customer feedback analysis, here are the key strengths of Atomberg's top competitors:
    {chr(10).join([f"**{brand.capitalize()}:** {', '.join([f'{point} ({count} mentions)' for point, count in [(point, competitor_plus_points[brand].count(point)) for point in list(set(competitor_plus_points[brand]))][:3]]) if competitor_plus_points[brand] else 'No specific strengths identified'}" for brand in top_competitors[:2]])}

    ### Atomberg YouTube Sentiment
    - **Transcript Sentiment:** {atomberg_transcript_sentiment}

    ### Keyword-Sentiment Associations
    - **Top Positive Keywords:** {sentiment_keyword_counts["positive"].most_common(5)}
    - **Top Negative Keywords:** {sentiment_keyword_counts["negative"].most_common(5)}
    
    ### Competitor Customer Feedback
    {competitor_insights}

    ### Detailed Sentiment Drivers (Raw Data)
    Here are some of the raw sentences containing keywords associated with positive and negative sentiment. Use these to provide detailed reasons, not just the keywords themselves.
    - **Positive Sentences:** {list(set(sentiment_reasons['positive']))[:5]}
    - **Negative Sentences:** {list(set(sentiment_reasons['negative']))[:5]}

    Your task is to provide a comprehensive analysis with the following sections:

    1. **Summary:** Provide a concise summary of Atomberg's overall performance and market position.

    2. **Competitive Landscape:** Analyze how Atomberg's metrics compare to its top competitors. What brand is the biggest threat and why?

    3. **Sentiment Drivers:** Based on the raw sentences provided, identify and explain the detailed reasons for positive and negative feedback. List the top 5 reasons for each.

    4. **YouTube Insights:** Analyze the sentiment of YouTube video transcripts. What does this suggest about the content being produced and its potential impact on the brand?

    5. **Actionable Recommendations (TOP 10 UNIQUE STRATEGIES):**
    
    Think from both marketing and economic perspectives. For each recommendation, provide an ultimate professional analysis, convincing why this strategy is the best. Explain in detail as much as possible (at least 7 to 8 sentences for every bullet on the following):
    - The specific strategy
    - How it increases SoV (Mentions Share, Engagement Share, Share of Positive Voice)
    - Economic impact (cost vs revenue potential)
    - Customer utility benefit
    - Step-by-step implementation flow

    Focus on innovative, feasible strategies that competitors haven't implemented yet.

    Each recommendation should be unique, implementable, and clearly explain the customer-company value exchange.

    Focus on innovative, feasible strategies that competitors haven't implemented yet. Each recommendation should be unique, implementable, and clearly explain the customer-company value exchange.
    """
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a highly skilled marketing strategist and business economist specializing in consumer electronics and brand positioning in the Indian market."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.85,
            max_tokens=8192
        )
        insights = response.choices[0].message.content.strip()
        print("\n=== Groq Insight Summary ===")
        print(insights)
    except Exception as e:
        print(f"[Groq error] {e}")

def save_insights_to_json(insights_data, filename=None):
    """Save insights data to a JSON file with timestamp."""
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"insights_report_{timestamp}.json"
    
    # Ensure the output directory exists
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(insights_data, f, indent=2, ensure_ascii=False)
        print(f"\n[✓] Insights saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"[x] Error saving insights to JSON: {e}")
        return None

if __name__ == "__main__":
    generate_insights(use_groq=True)