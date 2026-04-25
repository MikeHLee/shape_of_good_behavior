"""Quick API connectivity + extended thinking test. Never prints the key."""
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = anthropic.Anthropic()

print("Testing basic API connectivity (claude-sonnet-4-6)...")
try:
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
    )
    print(f"  Basic: OK -> {r.content[0].text!r}")
except Exception as e:
    print(f"  Basic: FAIL -> {e}")

print("Testing extended thinking (claude-sonnet-4-6)...")
try:
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=5000,
        thinking={"type": "enabled", "budget_tokens": 3000},
        messages=[{"role": "user", "content": "What is 2+2? Reason step by step."}],
    )
    blocks = [
        (b.type, getattr(b, "text", getattr(b, "thinking", ""))[:80])
        for b in r.content
    ]
    print(f"  Thinking: OK -> {blocks}")
except Exception as e:
    print(f"  Thinking: FAIL -> {e}")
