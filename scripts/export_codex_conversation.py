#!/usr/bin/env python3
"""Export a Codex JSONL session into readable Markdown and compact JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    chunks: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"input_text", "output_text", "text"}:
            text = item.get("text")
            if text:
                chunks.append(str(text))
        elif item_type in {"image", "input_image"}:
            chunks.append("[image]")
        elif item_type:
            chunks.append(f"[{item_type}]")
    return "\n".join(chunks).strip()


def load_messages(session_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []

    with session_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            row_type = row.get("type")
            payload = row.get("payload", {})
            if row_type == "session_meta":
                meta = payload
                continue

            if row_type != "response_item" or not isinstance(payload, dict):
                continue

            if payload.get("type") != "message":
                continue

            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue

            text = _text_from_content(payload.get("content"))
            if not text:
                continue

            messages.append(
                {
                    "line": line_no,
                    "timestamp": row.get("timestamp"),
                    "role": role,
                    "text": text,
                }
            )

    return meta, messages


def write_markdown(path: Path, session_path: Path, meta: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    title = meta.get("thread_name") or meta.get("id") or session_path.stem
    lines: list[str] = [
        f"# Codex Conversation Export - {title}",
        "",
        f"- Source: `{session_path}`",
        f"- Session ID: `{meta.get('id', '')}`",
        f"- CWD: `{meta.get('cwd', '')}`",
        f"- Messages: {len(messages)}",
        "",
    ]

    for idx, msg in enumerate(messages, 1):
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.extend(
            [
                f"## {idx}. {role}",
                "",
                f"Timestamp: `{msg.get('timestamp')}`  ",
                f"Source line: `{msg.get('line')}`",
                "",
                msg["text"].rstrip(),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/conversations"))
    args = parser.parse_args()

    session_path = args.session.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta, messages = load_messages(session_path)
    session_id = meta.get("id") or session_path.stem
    prefix = args.out_dir / f"codex_conversation_{session_id}"

    markdown_path = prefix.with_suffix(".md")
    json_path = prefix.with_suffix(".messages.json")
    raw_path = prefix.with_suffix(".raw.jsonl")

    write_markdown(markdown_path, session_path, meta, messages)
    json_path.write_text(
        json.dumps({"session": str(session_path), "meta": meta, "messages": messages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    raw_path.write_bytes(session_path.read_bytes())

    print(markdown_path)
    print(json_path)
    print(raw_path)
    print(f"messages={len(messages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
