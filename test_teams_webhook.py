"""
Quick check: is TEAMS_WEBHOOK_URL set, and does Teams accept a test message?

Usage (PowerShell):
  $env:TEAMS_WEBHOOK_URL = "https://...."
  python test_teams_webhook.py
"""
import sys
from pathlib import Path

BASE = str(Path(__file__).resolve().parent)
sys.path.insert(0, BASE)

from teams_executive_summary import resolve_teams_webhook_url, send_teams_payload


def main():
    url = resolve_teams_webhook_url()
    if not url:
        print("FAIL: No webhook URL found.")
        print("  Set TEAMS_WEBHOOK_URL in the environment, or create:")
        print("  .streamlit/secrets.toml  (copy from secrets.toml.example) with your URL.")
        sys.exit(1)
    print(f"OK: URL loaded ({len(url)} chars, starts with {url[:40]}...)")
    ok, msg = send_teams_payload(
        {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "0076D7",
            "summary": "IDS webhook test",
            "sections": [
                {
                    "activityTitle": "Webhook connectivity test",
                    "markdown": True,
                    "facts": [{"name": "Result", "value": "If you see this card, the webhook works."}],
                }
            ],
        }
    )
    if ok:
        print("OK: Message accepted by Teams (HTTP 200). Check your channel.")
    else:
        print(f"FAIL: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
