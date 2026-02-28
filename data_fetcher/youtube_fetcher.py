import os
import re
from googleapiclient.discovery import build
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL."""
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"embed/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")

def fetch_comments(video_url: str, max_comments: int = 500) -> pd.DataFrame:
    """
    Fetch up to max_comments top-level comments from a YouTube video.
    Returns a DataFrame with columns:
      author, text, likes, published_at, updated_at, author_channel_id
    """
    video_id = extract_video_id(video_url)
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_comments - len(comments)),
            pageToken=next_page_token,
            textFormat="plainText",
            order="time"  # "time" or "relevance"
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id":        item["id"],
                "author":            snippet.get("authorDisplayName", "Unknown"),
                "author_channel_id": snippet.get("authorChannelId", {}).get("value", ""),
                "text":              snippet.get("textDisplay", ""),
                "likes":             snippet.get("likeCount", 0),
                "published_at":      snippet.get("publishedAt", ""),
                "updated_at":        snippet.get("updatedAt", ""),
                "reply_count":       item["snippet"].get("totalReplyCount", 0)
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    df = pd.DataFrame(comments)
    # Convert timestamps to datetime objects
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["updated_at"]   = pd.to_datetime(df["updated_at"])
    return df

def fetch_video_metadata(video_url: str) -> dict:
    """Fetch title, view count, like count, comment count for the video."""
    video_id = extract_video_id(video_url)
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()
    if not response["items"]:
        return {}
    item = response["items"][0]
    return {
        "title":         item["snippet"]["title"],
        "channel":       item["snippet"]["channelTitle"],
        "view_count":    int(item["statistics"].get("viewCount", 0)),
        "like_count":    int(item["statistics"].get("likeCount", 0)),
        "comment_count": int(item["statistics"].get("commentCount", 0)),
        "published_at":  item["snippet"]["publishedAt"]
    }
