"""
Finds your Telegram chat ID and sends a test message, so the trading bot can message your phone.

Steps (all on your phone, in the Telegram app):
  1. Search for @BotFather, open it, send /newbot, and follow the questions.
     It gives you a TOKEN: a long text like 1234567890:AAH... Keep it private.
  2. Open the new bot you just made (BotFather gives you a link), press START,
     and send it any message, for example: hello
Then, on your PC, in Command Prompt:
  3. python telegram_setup.py YOUR_TOKEN
     It prints your Chat ID and sends a test message to your phone.
  4. Put the token and chat ID in the TELEGRAM_TOKEN and TELEGRAM_CHAT_ID lines near the top
     of demo_bot.py, inside the quotation marks.
Do not share the token with anyone, including in chats.
"""
import json
import sys
import urllib.parse
import urllib.request


def call(token, method, params=None):
    data = urllib.parse.urlencode(params).encode() if params else None
    with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/{method}", data=data, timeout=15) as r:
        return json.load(r)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python telegram_setup.py YOUR_TOKEN")
    token = sys.argv[1].strip()
    try:
        updates = call(token, "getUpdates")
    except Exception as e:
        sys.exit(f"Could not reach Telegram, or the token is wrong: {e}")
    chats = {}
    for u in updates.get("result", []):
        m = u.get("message") or u.get("channel_post") or {}
        c = m.get("chat")
        if c:
            chats[c["id"]] = c.get("first_name") or c.get("title") or ""
    if not chats:
        sys.exit("No messages found. Open your bot in Telegram, press START, send it 'hello', "
                 "then run this again.")
    for chat_id, name in chats.items():
        print(f"Chat ID: {chat_id}  ({name})")
        call(token, "sendMessage", {"chat_id": chat_id, "text": "Test message from your trading bot setup. It works!"})
    print("A test message was sent to your phone. Put the token and this chat ID into demo_bot.py.")


if __name__ == "__main__":
    main()
