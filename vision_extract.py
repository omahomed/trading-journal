"""
MarketSurge Screenshot Data Extraction via Claude Vision API
Extracts fundamental data from MarketSurge stock analysis screenshots.
"""

import anthropic
import streamlit as st
import base64
import json
from typing import Optional, Dict


def get_anthropic_client():
    """Initialize Anthropic client from Streamlit secrets."""
    try:
        api_key = st.secrets.get("anthropic", {}).get("api_key")
        if not api_key:
            return None
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        print(f"[Vision] Failed to init Anthropic client: {e}")
        return None


EXTRACTION_PROMPT = """Analyze this MarketSurge stock screenshot and extract the following fundamental data fields.
Return ONLY a valid JSON object with these keys. Use null for any field not visible or not readable.

Required fields:
{
    "ticker": "stock ticker symbol",
    "composite_rating": integer (1-99),
    "eps_rating": integer (1-99),
    "rs_rating": integer (1-99),
    "group_rs_rating": string (e.g. "A+", "B", "C-"),
    "smr_rating": string (e.g. "A", "B", "N/A"),
    "acc_dis_rating": string (e.g. "A+", "B-", "C"),
    "timeliness_rating": string,
    "sponsorship_rating": string (e.g. "A", "B"),
    "eps_growth_rate": number (percentage, e.g. 11 for 11%),
    "ud_vol_ratio": number (e.g. 1.1),
    "annual_eps": [{"year": "2022", "eps": 5.23}, ...],
    "quarterly_eps": [{"quarter": "Mar 24", "eps": 1.52, "change_pct": 15}, ...],
    "quarterly_sales": [{"quarter": "Mar 24", "sales_mil": 1234, "change_pct": 8}, ...],
    "mgmt_own_pct": number or null,
    "banks_own_pct": number or null,
    "funds_own_pct": number or null,
    "num_funds": integer or null,
    "price": number or null,
    "market_cap": string or null,
    "industry_group": string or null,
    "industry_group_rank": integer or null
}

Important:
- Extract exactly what you see. Do not guess or infer missing values.
- For ratings like "N/A" or "--", return the string "N/A".
- For percentage fields, return just the number (e.g. 11 not "11%").
- Return ONLY the JSON object, no markdown formatting, no explanation."""


def extract_fundamentals(image_bytes: bytes, file_name: str = "image.png") -> Optional[Dict]:
    """
    Extract fundamental data from a MarketSurge screenshot using Claude Vision.

    Args:
        image_bytes: Raw image bytes
        file_name: Original filename (used to determine media type)

    Returns:
        Dictionary of extracted fundamentals, or None if failed
    """
    client = get_anthropic_client()
    if not client:
        return None

    # Determine media type
    ext = file_name.lower().split('.')[-1] if '.' in file_name else 'png'
    media_type_map = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'webp': 'image/webp'
    }
    media_type = media_type_map.get(ext, 'image/png')

    # Encode image to base64
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": EXTRACTION_PROMPT
                        }
                    ]
                }
            ]
        )

        # Parse response
        raw_text = response.content[0].text.strip()

        # Clean potential markdown wrapping
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()

        data = json.loads(raw_text)
        return data

    except json.JSONDecodeError as e:
        print(f"[Vision] Failed to parse JSON response: {e}")
        print(f"[Vision] Raw response: {raw_text[:500]}")
        return None
    except anthropic.APIError as e:
        print(f"[Vision] Anthropic API error: {e}")
        return None
    except Exception as e:
        print(f"[Vision] Extraction failed: {e}")
        return None


def is_available() -> bool:
    """Check if Vision API is configured and available."""
    api_key = st.secrets.get("anthropic", {}).get("api_key")
    return bool(api_key)
