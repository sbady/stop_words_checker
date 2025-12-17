import argparse
import html
import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup, Tag


DEFAULT_STOP_WORDS = ["deprecated", "legacy", "removed", "unsupported"]
HEADERS = {"User-Agent": "DocStopwordScanner/1.0 (+https://example.com)"}


@dataclass
class Hit:
    word: str
    match: str
    context: str
    index: int


@dataclass
class StopWordRule:
    raw: str
    key: str
    pattern: re.Pattern
    allow_patterns: List[re.Pattern] = field(default_factory=list)


@dataclass
class PageReport:
    url: str
    title: str
    hits_by_word: Dict[str, List[Hit]] = field(default_factory=dict)

    @property
    def total_hits(self) -> int:
        return sum(len(items) for items in self.hits_by_word.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursive documentation crawler that highlights stop-words."
    )
    parser.add_argument(
        "--url",
        help="Стартовый URL раздела документации (например, https://site.com/docs/).",
    )
    parser.add_argument(
        "--stop-words",
        dest="stop_words_path",
        help="Путь к файлу со стоп-словами (по одному на строку; поддерживаются allow-исключения). "
        "Если не указан, используется встроенный список.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Задержка между запросами в секундах (по умолчанию 0.5).",
    )
    parser.add_argument(
        "--output",
        help="Путь к файлу для сохранения полного отчета в формате JSON.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Необязательно: ограничить количество посещаемых страниц.",
    )
    parser.add_argument(
        "--env",
        dest="env_path",
        default=".env",
        help="Путь к .env файлу с параметрами URL и STOP_WORDS_FILE (по умолчанию .env).",
    )
    return parser.parse_args()


def load_stop_words(path: Optional[str]) -> List[StopWordRule]:
    """
    Load stop-word rules from file (supports `allow:` исключения).
    """
    return load_stop_word_rules(path)


def load_stop_words_raw(path: Optional[str]) -> List[str]:
    """
    Load ONLY raw stop-words (strings). `allow:` строки не возвращаются.
    """
    return [rule.raw for rule in load_stop_word_rules(path)]


def _compile_pattern(raw: str) -> Tuple[str, re.Pattern]:
    word = raw.strip()
    if word.lower().startswith("re:"):
        regex = word[3:].strip()
        if not regex:
            raise ValueError("Пустой regex после 're:'")
        # Stop-words file is plain text, but people often write escapes as "\\w" / "\\s".
        # Make it tolerant by unescaping double backslashes.
        regex = regex.replace("\\\\", "\\")
        return word, re.compile(regex, re.IGNORECASE)
    return word.lower(), re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


def _compile_allow_pattern(raw: str) -> re.Pattern:
    value = raw.strip()
    if value.lower().startswith("re:"):
        regex = value[3:].strip()
        if not regex:
            raise ValueError("Пустой allow-regex после 're:'")
        regex = regex.replace("\\\\", "\\")
        return re.compile(regex, re.IGNORECASE)
    return re.compile(re.escape(value), re.IGNORECASE)


def load_stop_word_rules(path: Optional[str]) -> List[StopWordRule]:
    """
    Формат файла стоп-слов:
    - 1 стоп-слово/фраза на строку
    - пустые строки и строки, начинающиеся с #, игнорируются
    - regex: строка начинается с "re:"
    - исключения (допустимые контексты) для предыдущего стоп-слова:
        allow: <фраза>
        allow: re:<regex>
      allow применяется к совпадениям, которые пересекаются с allow-шаблоном
      в пределах небольшого окна вокруг найденного стоп-слова.
    """
    if not path:
        return [_rule_from_raw(w) for w in DEFAULT_STOP_WORDS]
    try:
        rules: List[StopWordRule] = []
        last_rule: Optional[StopWordRule] = None
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    continue
                if stripped.lower().startswith("allow:"):
                    if not last_rule:
                        print(
                            f"[WARN] allow без стоп-слова в файле: {path}",
                            file=sys.stderr,
                        )
                        continue
                    allow_value = stripped.split(":", 1)[1].strip()
                    if not allow_value:
                        continue
                    try:
                        last_rule.allow_patterns.append(_compile_allow_pattern(allow_value))
                    except (ValueError, re.error) as exc:
                        print(
                            f"[WARN] Некорректный allow '{allow_value}' для '{last_rule.raw}': {exc}",
                            file=sys.stderr,
                        )
                    continue

                last_rule = _rule_from_raw(stripped)
                rules.append(last_rule)

        return rules or [_rule_from_raw(w) for w in DEFAULT_STOP_WORDS]
    except OSError as exc:
        print(f"Не удалось прочитать файл стоп-слов: {exc}", file=sys.stderr)
        return [_rule_from_raw(w) for w in DEFAULT_STOP_WORDS]


def _rule_from_raw(raw: str) -> StopWordRule:
    try:
        key, pattern = _compile_pattern(raw)
    except (ValueError, re.error) as exc:
        print(f"[WARN] Некорректное стоп-слово '{raw}': {exc}", file=sys.stderr)
        key, pattern = _compile_pattern("re:(?!)")  # never matches
    return StopWordRule(raw=raw.strip(), key=key, pattern=pattern)


def load_env_file(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    # Упрощенный формат: если строка выглядит как URL, считаем ее URL=...
                    if stripped.startswith(("http://", "https://")) and "URL" not in data:
                        data["URL"] = stripped
                    continue
                key, value = stripped.split("=", 1)
                data[key.strip()] = value.strip()
    except FileNotFoundError:
        return data
    except OSError as exc:
        print(f"[WARN] Не удалось прочитать {path}: {exc}", file=sys.stderr)
    return data


def normalize_url(raw_url: str) -> Optional[str]:
    parsed = urlsplit(raw_url)
    if parsed.scheme not in ("http", "https"):
        return None

    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    normalized = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return normalized


def scope_path(parsed_start) -> str:
    """
    Determine the path prefix that defines the crawling scope.
    Example: /docs/guide -> /docs/
    """
    start_path = parsed_start.path or "/"
    if not start_path.endswith("/"):
        start_path = start_path.rsplit("/", 1)[0] + "/"
    return start_path


def is_probably_html(url: str) -> bool:
    path = urlsplit(url).path.lower()
    blocked_suffixes = (
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".zip",
        ".tar",
        ".gz",
        ".mp4",
        ".mp3",
    )
    return not any(path.endswith(ext) for ext in blocked_suffixes)


def fetch_page(url: str, session: requests.Session) -> Optional[requests.Response]:
    try:
        response = session.get(url, headers=HEADERS, timeout=10)
        if response.status_code >= 400:
            print(f"[WARN] Пропуск {url} — статус {response.status_code}", file=sys.stderr)
            return None
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            return None
        return response
    except requests.RequestException as exc:
        print(f"[WARN] Ошибка при загрузке {url}: {exc}", file=sys.stderr)
        return None


def extract_title(soup: BeautifulSoup, diplodoc_data: Optional[dict] = None) -> str:
    if diplodoc_data:
        title = diplodoc_data.get("data", {}).get("title")
        if title:
            return str(title).strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return "Без названия"


def choose_main_container(soup: BeautifulSoup) -> Tag:
    selectors = [
        "article",
        "main",
        "div[role=main]",
        "section[role=main]",
        "div#content",
        "div.content",
        "div#main",
        "div.main",
    ]
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            return element
    return soup.body or soup


def extract_diplodoc_fragment(soup: BeautifulSoup) -> Tuple[Optional[Tag], Optional[dict]]:
    """
    Diplodoc статика кладет контент в <script id="diplodoc-state">.
    Варианты:
    - поле data.html: готовый HTML статьи;
    - поле data.data.blocks: список карточек/секций с html-текстом в полях title/description/text.
    """
    script = soup.find("script", id="diplodoc-state")
    if not script or not script.string:
        return None, None
    try:
        data = json.loads(script.string)
    except json.JSONDecodeError:
        return None, None
    html_fragment = data.get("data", {}).get("html")
    if not html_fragment:
        return None, data
    unescaped = html.unescape(html_fragment)
    fragment_soup = BeautifulSoup(unescaped, "html.parser")
    return fragment_soup, data


def collect_diplodoc_blocks_text(data: dict) -> Optional[str]:
    blocks = data.get("data", {}).get("data", {}).get("blocks", [])
    if not blocks:
        return None

    def gather(block_list: List[dict], acc: List[str]) -> None:
        for blk in block_list:
            for key in ("title", "description", "text"):
                val = blk.get(key)
                if val:
                    acc.append(html.unescape(str(val)))
            if "children" in blk and isinstance(blk["children"], list):
                gather(blk["children"], acc)
            if "items" in blk and isinstance(blk["items"], list):
                gather(blk["items"], acc)

    parts: List[str] = []
    gather(blocks, parts)
    if not parts:
        return None
    text = BeautifulSoup(" ".join(parts), "html.parser").get_text(separator=" ", strip=True)
    return text


def parse_toc_links(start_url: str, session: requests.Session) -> List[str]:
    """
    Для Diplodoc есть toc.js с деревом ссылок. Используем его как источник всех статей,
    чтобы не зависеть от наличия ссылок на страницах.
    """
    parsed_start = urlsplit(start_url)
    base_for_toc = start_url if start_url.endswith("/") else f"{start_url}/"
    toc_url = urljoin(base_for_toc, "toc.js")
    site_root = f"{parsed_start.scheme}://{parsed_start.netloc}/"

    try:
        resp = session.get(toc_url, headers=HEADERS, timeout=10)
        if resp.status_code >= 400:
            return []
        content = resp.text
    except requests.RequestException:
        return []

    start = content.find("=")
    if start == -1:
        return []
    json_part = content[start + 1 :].strip()
    if json_part.endswith(";"):
        json_part = json_part[:-1]

    try:
        toc_data = json.loads(json_part)
    except json.JSONDecodeError:
        return []

    links: List[str] = []

    def walk(node: dict) -> None:
        href = node.get("href")
        if href:
            links.append(urljoin(site_root, href))
        for child_list_key in ("items", "children"):
            children = node.get(child_list_key)
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, dict):
                        walk(child)

    walk(toc_data)
    return links


def extract_text(element: Tag) -> str:
    return element.get_text(separator=" ", strip=True)


def extract_links(element: Tag, base_url: str) -> List[str]:
    links: List[str] = []
    for anchor in element.find_all("a", href=True):
        href = anchor["href"]
        absolute = urljoin(base_url, href)
        normalized = normalize_url(absolute)
        if normalized:
            links.append(normalized)
    return links


def find_stopword_hits(
    text: str, stop_words: Iterable[StopWordRule]
) -> Dict[str, List[Hit]]:
    def hit_is_allowed(match_start: int, match_end: int, allow_patterns: List[re.Pattern]) -> bool:
        if not allow_patterns:
            return False
        window = 80
        slice_start = max(0, match_start - window)
        slice_end = min(len(text), match_end + window)
        snippet = text[slice_start:slice_end]
        for allow_pattern in allow_patterns:
            for allow_match in allow_pattern.finditer(snippet):
                allow_start = slice_start + allow_match.start()
                allow_end = slice_start + allow_match.end()
                if allow_start < match_end and allow_end > match_start:
                    return True
        return False

    hits: Dict[str, List[Hit]] = {}
    for rule in stop_words:
        for idx, match in enumerate(rule.pattern.finditer(text), start=1):
            if hit_is_allowed(match.start(), match.end(), rule.allow_patterns):
                continue
            context = extract_context(text, match.start(), match.end())
            hits.setdefault(rule.key, []).append(
                Hit(word=rule.raw, match=match.group(0), context=context, index=idx)
            )
    return hits


def extract_context(
    text: str,
    start: int,
    end: int,
    window: int = 80,
    max_len: int = 280,
    scan_limit: int = 500,
) -> str:
    """
    Extract a readable context snippet around the match.

    1) Try to capture "sentence-like" text between delimiters, but only within scan_limit.
    2) If the sentence is too long (or delimiters are missing), fallback to a fixed window.
    3) Always cap the returned snippet length to max_len.
    """
    sentence_delimiters = ".!?\n\r…;"

    left_scan_start = max(0, start - scan_limit)
    right_scan_end = min(len(text), end + scan_limit)

    left_part = text[left_scan_start:start]
    right_part = text[end:right_scan_end]

    left_rel = max(left_part.rfind(d) for d in sentence_delimiters)
    right_positions = [right_part.find(d) for d in sentence_delimiters]
    right_positions = [p for p in right_positions if p != -1]

    left_bound = (left_scan_start + left_rel) if left_rel != -1 else -1
    right_bound = (end + min(right_positions)) if right_positions else -1

    def window_snippet() -> str:
        left = max(0, start - window)
        right = min(len(text), end + window)
        return text[left:right].strip()

    if left_bound != -1 or right_bound != -1:
        left = left_bound + 1 if left_bound != -1 else 0
        right = right_bound if right_bound != -1 else len(text)
        snippet = text[left:right].strip()
        if len(snippet) > max_len:
            snippet = window_snippet()
    else:
        snippet = window_snippet()

    snippet = " ".join(snippet.split())
    if len(snippet) > max_len:
        center_left = max(0, (start - max_len // 2))
        center_right = min(len(text), center_left + max_len)
        snippet = " ".join(text[center_left:center_right].strip().split())
        if center_left > 0:
            snippet = "..." + snippet
        if center_right < len(text):
            snippet = snippet + "..."

    return snippet


def crawl(
    start_url: str, stop_words: List[StopWordRule], delay: float, max_pages: Optional[int]
) -> Tuple[List[PageReport], List[str]]:
    parsed_start = urlsplit(start_url)
    base_domain = parsed_start.netloc
    base_scope = scope_path(parsed_start)
    queue: Deque[str] = deque()
    visited: Set[str] = set()
    reports: List[PageReport] = []
    failed: List[str] = []
    session = requests.Session()

    normalized_start = normalize_url(start_url)
    if not normalized_start:
        print("Стартовый URL должен использовать http или https.", file=sys.stderr)
        return reports, failed
    queue.append(normalized_start)

    toc_links = parse_toc_links(normalized_start, session)
    use_toc_links = bool(toc_links)
    if use_toc_links:
        for toc_link in toc_links:
            normalized_link = normalize_url(toc_link)
            if (
                normalized_link
                and urlsplit(normalized_link).netloc == base_domain
                and normalized_link not in queue
            ):
                queue.append(normalized_link)

    while queue:
        if max_pages and len(visited) >= max_pages:
            break

        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)

        print(f"Сканируем: {current}")
        response = fetch_page(current, session)
        if not response:
            failed.append(current)
            time.sleep(delay)
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        diplodoc_fragment, diplodoc_data = extract_diplodoc_fragment(soup)
        diplodoc_blocks_text = collect_diplodoc_blocks_text(diplodoc_data or {})
        is_diplodoc = diplodoc_data is not None

        if diplodoc_fragment:
            main_container = diplodoc_fragment
            text = extract_text(main_container)
        elif diplodoc_blocks_text:
            main_container = None
            text = diplodoc_blocks_text
        else:
            main_container = choose_main_container(soup)
            text = extract_text(main_container)

        title = extract_title(soup, diplodoc_data)
        hits = find_stopword_hits(text, stop_words)
        reports.append(PageReport(url=current, title=title, hits_by_word=hits))

        if main_container and not is_diplodoc and not use_toc_links:
            for link in extract_links(main_container, current):
                parsed_link = urlsplit(link)
                if parsed_link.netloc != base_domain:
                    continue
                if not parsed_link.path.startswith(base_scope):
                    continue
                if not is_probably_html(link):
                    continue
                if link not in visited and link not in queue:
                    queue.append(link)

        time.sleep(delay)

    return reports, failed


def print_report(start_url: str, reports: List[PageReport], failed: List[str]) -> None:
    total_pages = len(reports)
    pages_with_hits = [r for r in reports if r.total_hits > 0]
    total_hits = sum(r.total_hits for r in pages_with_hits)

    print(f"Начинаем обход с {start_url}")
    print(f"Проверено страниц: {total_pages}")
    print(f"Найдено проблемных статей: {len(pages_with_hits)}")
    print("==================================")

    for report in pages_with_hits:
        print()
        print(f'Статья: "{report.title}"')
        print(f"URL: {report.url}")
        print(f"Всего стоп-слов найдено: {report.total_hits}")

        for word, hits in sorted(
            report.hits_by_word.items(), key=lambda item: len(item[1]), reverse=True
        ):
            print(f'\nСлово "{word}" ({len(hits)} вхождений):')
            for hit in hits:
                print(f"  - {hit.context}")

        print("\n---")

    if failed:
        print("\nНе удалось обработать (пропущены):")
        for url in failed:
            print(f" - {url}")

    print("==================================")
    print(
        f"ИТОГО: В {len(pages_with_hits)} из {total_pages} статей найдены стоп-слова. "
        f"Всего вхождений: {total_hits}."
    )


def export_json(path: str, reports: List[PageReport], start_url: str) -> None:
    payload = {
        "start_url": start_url,
        "pages_checked": len(reports),
        "total_hits": sum(r.total_hits for r in reports),
        "pages": [],
    }
    for report in reports:
        payload["pages"].append(
            {
                "url": report.url,
                "title": report.title,
                "total_hits": report.total_hits,
                "hits": {
                    word: [
                        {"index": hit.index, "match": hit.match, "context": hit.context}
                        for hit in hits
                    ]
                    for word, hits in report.hits_by_word.items()
                },
            }
        )
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"JSON-отчет сохранен в {path}")
    except OSError as exc:
        print(f"Не удалось записать JSON-отчет: {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    env_values = load_env_file(args.env_path)

    start_url = args.url or env_values.get("URL")
    stop_words_path = args.stop_words_path or env_values.get("STOP_WORDS_FILE")

    if not start_url:
        print(
            "Укажите стартовый URL через --url или переменную URL в .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    stop_words = load_stop_words(stop_words_path)
    reports, failed = crawl(
        start_url=start_url,
        stop_words=stop_words,
        delay=args.delay,
        max_pages=args.max_pages,
    )
    if not reports:
        print("Не удалось обработать ни одной страницы.")
        return

    print_report(start_url, reports, failed)

    if args.output:
        export_json(args.output, reports, start_url)


if __name__ == "__main__":
    main()
