"""把 Claude Code 的 JSONL transcript 匯出成可讀 markdown。

只保留 user / assistant 的文字，濾掉工具呼叫、工具輸出、思考、圖片、
以及 slash 指令的系統殘留，產出乾淨的對話記錄。

用法：
  python chat_record.py                 # 自動抓「最新的」對話（通常就是目前這場）
  python chat_record.py <檔.jsonl>      # 指定某場對話
  python chat_record.py -o 名稱.md      # 自訂輸出檔名
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

# Windows 終端預設 cp950 會讓 emoji/中文 print 崩潰，統一改用 UTF-8 輸出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 本專案的 transcript 都存在這裡（每場對話一個 .jsonl）
PROJECT_DIR = os.path.expanduser(r"~/.claude/projects/d--ray")

# slash 指令會在 user 訊息留下這些系統標記，匯出時濾掉
_NOISE_MARKERS = (
    "<local-command-caveat>", "<command-name>", "<command-message>",
    "<local-command-stdout>", "Caveat: The messages below",
)


def latest_transcript():
    """回傳專案資料夾中『最近修改』的 .jsonl —— 通常就是目前這場對話。"""
    files = glob.glob(os.path.join(PROJECT_DIR, "*.jsonl"))
    if not files:
        raise FileNotFoundError(f"找不到任何對話檔於 {PROJECT_DIR}")
    return max(files, key=os.path.getmtime)


def extract_text(content):
    """從 message.content 取純文字（content 可能是 str 或 block list）。"""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "").strip()
            if t:
                parts.append(t)
    return "\n\n".join(parts)


def is_noise(text):
    return any(m in text for m in _NOISE_MARKERS)


def export_to_markdown(jsonl_file, output_file, keep_noise=False):
    if not os.path.exists(jsonl_file):
        print(f"找不到檔案：{jsonl_file}")
        return

    rows = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = record.get("message")
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue

            content = extract_text(msg.get("content"))
            if not content:
                continue
            if not keep_noise and role == "user" and is_noise(content):
                continue
            rows.append((role, content))

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# Claude Code 對話紀錄\n\n")
        out.write(f"> 匯出時間：{datetime.now():%Y-%m-%d %H:%M:%S}　"
                  f"來源：`{os.path.basename(jsonl_file)}`　"
                  f"訊息數：{len(rows)}\n\n---\n\n")
        for role, content in rows:
            who = "🧑 User" if role == "user" else "🤖 Claude"
            out.write(f"### {who}\n\n{content}\n\n---\n\n")

    print(f"✅ 匯出成功！{len(rows)} 則訊息 → {output_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Claude Code 對話 → markdown")
    ap.add_argument("jsonl", nargs="?", default=None,
                    help="對話 .jsonl 路徑（省略=自動抓最新那場）")
    ap.add_argument("-o", "--out", default="exported_chat.md",
                    help="輸出 markdown 檔名（預設 exported_chat.md）")
    ap.add_argument("--keep-noise", action="store_true",
                    help="保留 slash 指令的系統殘留（預設濾掉）")
    args = ap.parse_args()

    src = args.jsonl or latest_transcript()
    print(f"來源：{src}")
    export_to_markdown(src, args.out, keep_noise=args.keep_noise)
