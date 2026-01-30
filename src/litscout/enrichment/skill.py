from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .models import ResearchGoal, SkillProfile


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*", re.DOTALL)


def _parse_inline_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        items = {}
        for part in inner.split(","):
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            items[k.strip()] = _parse_inline_value(v)
        return items
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_inline_value(v) for v in inner.split(",")]
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"')


def _parse_yaml(text: str) -> dict[str, Any]:
    lines = [line.rstrip() for line in text.splitlines()]

    def next_nonempty(idx: int) -> int:
        while idx < len(lines):
            stripped = lines[idx].strip()
            if stripped and not stripped.startswith("#"):
                return idx
            idx += 1
        return idx

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def parse_block(idx: int, indent: int) -> tuple[Any, int]:
        idx = next_nonempty(idx)
        if idx >= len(lines):
            return {}, idx
        if indent_of(lines[idx]) < indent:
            return {}, idx

        stripped = lines[idx].lstrip()
        if stripped.startswith("- "):
            items: list[Any] = []
            while idx < len(lines):
                idx = next_nonempty(idx)
                if idx >= len(lines):
                    break
                line = lines[idx]
                if indent_of(line) < indent:
                    break
                stripped = line.lstrip()
                if not stripped.startswith("- "):
                    break
                item_text = stripped[2:].strip()
                if item_text == "":
                    value, idx = parse_block(idx + 1, indent + 2)
                else:
                    value = _parse_inline_value(item_text)
                    idx += 1
                items.append(value)
            return items, idx

        mapping: dict[str, Any] = {}
        while idx < len(lines):
            idx = next_nonempty(idx)
            if idx >= len(lines):
                break
            line = lines[idx]
            if indent_of(line) < indent:
                break
            stripped = line.lstrip()
            if ":" not in stripped:
                raise ValueError(f"invalid line: {line}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                child, idx = parse_block(idx + 1, indent_of(line) + 2)
                mapping[key] = child
            else:
                mapping[key] = _parse_inline_value(value)
                idx += 1
        return mapping, idx

    result, _ = parse_block(0, 0)
    if not isinstance(result, dict):
        raise ValueError("top-level YAML must be a mapping")
    return result


def _load_yaml_from_text(text: str) -> dict[str, Any]:
    front = _FRONTMATTER_RE.match(text)
    if front:
        text = front.group(1)
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        return json.loads(text)
    if "# Research Goals" in text:
        prefix, _ = text.split("# Research Goals", 1)
        text = prefix.strip()
        if not text:
            return {}
    return _parse_yaml(text)


def _hash_skill(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_skill_profile(path: str) -> SkillProfile:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    raw = _load_yaml_from_text(text)
    profile = SkillProfile.model_validate(raw)
    goals = _parse_goals(text)
    if goals:
        profile.goals = goals
    canonical = profile.model_dump(exclude={"skill_hash"}, by_alias=True)
    profile.skill_hash = _hash_skill(canonical)
    return profile


_GOAL_HEADER_RE = re.compile(r"^#{2,3}\s+(G\d+)\s*[:：]\s*(.+)$")


def _parse_goals(text: str) -> list[ResearchGoal]:
    lines = text.splitlines()
    goals: list[ResearchGoal] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        match = _GOAL_HEADER_RE.match(line)
        if not match:
            idx += 1
            continue
        goal_id = match.group(1).strip()
        name = match.group(2).strip()
        idx += 1

        description = ""
        signals: list[str] = []
        negative_signals: list[str] = []
        current_key = None
        while idx < len(lines):
            raw = lines[idx]
            stripped = raw.strip()
            if _GOAL_HEADER_RE.match(stripped):
                break
            if stripped.startswith("description:") or stripped in ("**描述**", "描述", "描述:"):
                current_key = "description"
                if stripped.endswith("|") or stripped in ("**描述**", "描述", "描述:"):
                    description = ""
                else:
                    description = stripped.split(":", 1)[1].strip()
                idx += 1
                while idx < len(lines):
                    raw2 = lines[idx]
                    if _GOAL_HEADER_RE.match(raw2.strip()) or raw2.strip().startswith(
                        ("signals:", "negative_signals:", "**正向信号", "**负向信号")
                    ):
                        break
                    if current_key == "description" and raw2.strip():
                        description += (raw2.rstrip() + "\n")
                    idx += 1
                description = description.strip()
                continue
            if stripped.startswith("signals:") or stripped in ("**正向信号**", "正向信号", "正向信号:"):
                current_key = "signals"
                idx += 1
                while idx < len(lines):
                    raw2 = lines[idx]
                    stripped2 = raw2.strip()
                    if _GOAL_HEADER_RE.match(stripped2) or stripped2.startswith(
                        ("negative_signals:", "description:", "**负向信号", "**描述**")
                    ):
                        break
                    if stripped2.startswith("- "):
                        signals.append(stripped2[2:].strip().strip("\""))
                    idx += 1
                continue
            if stripped.startswith("negative_signals:") or stripped in ("**负向信号**", "负向信号", "负向信号:"):
                current_key = "negative_signals"
                idx += 1
                while idx < len(lines):
                    raw2 = lines[idx]
                    stripped2 = raw2.strip()
                    if _GOAL_HEADER_RE.match(stripped2) or stripped2.startswith(
                        ("signals:", "description:", "**正向信号", "**描述**")
                    ):
                        break
                    if stripped2.startswith("- "):
                        negative_signals.append(stripped2[2:].strip().strip("\""))
                    idx += 1
                continue
            idx += 1

        if not description:
            raise ValueError(f"Goal {goal_id} missing description")
        goals.append(
            ResearchGoal(
                goal_id=goal_id,
                name=name,
                description=description,
                signals=signals,
                negative_signals=negative_signals,
            )
        )
    return goals
