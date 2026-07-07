"""LINE公式アカウントへブロードキャスト通知を送るCLI(GitHub Actionsから利用)

友だち追加した本人にだけ届く前提でbroadcast APIを使う(1:1の相手のuser IDを
取得する仕組み-Webhook受信-を組まずに済むため)。

環境変数 LINE_CHANNEL_ACCESS_TOKEN が必要。未設定なら何もせず正常終了する
(ローカル実行やトークン未設定時に落とさないため)。

    python notify_line.py "送信するテキスト"
"""
import json
import os
import sys
import urllib.request

API_URL = "https://api.line.me/v2/bot/message/broadcast"


def send(text: str) -> bool:
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        print("LINE_CHANNEL_ACCESS_TOKEN が未設定のため通知をスキップします")
        return False

    payload = json.dumps(
        {"messages": [{"type": "text", "text": text}]}, ensure_ascii=False
    ).encode("utf-8")
    req = urllib.request.Request(
        API_URL, data=payload, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print("LINE通知を送信しました (status", resp.status, ")")
            return True
    except urllib.error.HTTPError as e:
        print("LINE通知に失敗しました:", e.code, e.read().decode(errors="replace"))
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python notify_line.py <text>")
        sys.exit(1)
    if not send(sys.argv[1]):
        sys.exit(1)
