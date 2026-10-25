"""Check the glossary's local links, index coverage, and document structure."""

import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
SECTIONS = ("## 개발할 때", "## 기억할 점", "## 함께 읽기")


def main():
    errors = []
    entries = sorted((ROOT / "explanations").glob("*.md"))
    documents = [ROOT / "README.md", ROOT / "AUTHORING.md", *entries]
    titles = {}
    contents = {}

    for path in documents:
        content = path.read_text(encoding="utf-8")
        # Code fences can contain documentation examples rather than real links.
        prose = re.sub(r"^```[^\n]*\n.*?^```[^\n]*$", "", content,
                       flags=re.MULTILINE | re.DOTALL)
        contents[path] = prose
        headings = re.findall(r"^# (.+)$", prose, re.MULTILINE)
        relative = path.relative_to(ROOT)
        if len(headings) != 1 or not prose.startswith("# "):
            errors.append(f"{relative}: expected one H1 at the start")
        elif path in entries:
            titles[path] = headings[0]

        for label, target in LINK.findall(prose):
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            resolved = (path.parent / unquote(url.path)).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.is_file():
                errors.append(f"{relative}: broken local link: {target}")

        if path in entries:
            offsets = [prose.find(section) for section in SECTIONS]
            if any(prose.count(section) != 1 for section in SECTIONS):
                errors.append(f"{relative}: expected each required section once")
            elif offsets != sorted(offsets):
                errors.append(f"{relative}: sections are out of order")
            related = prose.partition("## 함께 읽기\n")[2]
            for label, target in LINK.findall(related):
                url = urlsplit(target)
                resolved = (path.parent / unquote(url.path)).resolve()
                if url.scheme or url.netloc or resolved not in entries or resolved == path:
                    errors.append(f"{relative}: invalid related entry: {target}")
            if not LINK.search(related):
                errors.append(f"{relative}: missing related entries")

    readme = contents[ROOT / "README.md"]
    index = readme.partition("## 목차\n")[2].partition("\n## ")[0]
    if not index:
        errors.append("README.md: missing glossary index")
    indexed = []
    for label, target in LINK.findall(index):
        resolved = (ROOT / unquote(urlsplit(target).path)).resolve()
        indexed.append(resolved)
        if resolved not in entries:
            errors.append(f"README.md: unexpected index target: {target}")
        elif titles.get(resolved) != label:
            errors.append(f"README.md: index label differs from H1: {target}")
    counts = Counter(indexed)
    for path in entries:
        if counts[path] != 1:
            errors.append(f"README.md: {path.name} appears {counts[path]} times in index")
    for count in re.findall(r"(\d+)개\s*용어", readme):
        if int(count) != len(entries):
            errors.append(f"README.md: says {count} entries, found {len(entries)}")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"OK: {len(entries)} entries; local links, index, titles, and structure checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
