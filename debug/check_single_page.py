import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stopword_crawler import (
    extract_diplodoc_fragment,
    collect_diplodoc_blocks_text,
    extract_text,
    extract_title,
    fetch_page,
    find_stopword_hits,
    load_env_file,
    load_stop_words,
    normalize_url,
    choose_main_container,
)

from bs4 import BeautifulSoup
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug tool: check a single Diplodoc/HTML page for stop-words."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Относительный путь от ru (например, about.html или documentation/first_steps/reg.html) "
        "ИЛИ полный URL.",
    )
    parser.add_argument(
        "--stop-words",
        default="debug/test_stopwords.txt",
        help="Путь к тестовому файлу стоп-слов (по умолчанию debug/test_stopwords.txt).",
    )
    parser.add_argument(
        "--base-url",
        help="Базовый URL (например, https://docs.wecloud.events/ru/). "
        "Если не указан, берётся из .env (URL=...).",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Путь к .env (по умолчанию .env). Используется для URL, если --base-url не задан.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=100,
        help="Сколько первых символов текста показать (по умолчанию 100).",
    )
    return parser.parse_args()


def build_target_url(path_or_url: str, base_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url

    base = base_url if base_url.endswith("/") else base_url + "/"
    site_root = f"{urlsplit(base).scheme}://{urlsplit(base).netloc}/"
    rel = path_or_url.lstrip("/")

    if rel.startswith("ru/"):
        return urljoin(site_root, rel)
    return urljoin(base, rel)


def main() -> None:
    args = parse_args()
    env = load_env_file(args.env)
    base_url = args.base_url or env.get("URL")
    if not base_url:
        print("Укажите --base-url или URL в .env.", file=sys.stderr)
        sys.exit(1)

    target_url = normalize_url(build_target_url(args.path, base_url))
    if not target_url:
        print("Некорректный URL/путь.", file=sys.stderr)
        sys.exit(1)

    stop_words = load_stop_words(args.stop_words)
    session = requests.Session()
    response = fetch_page(target_url, session)
    if not response:
        print(f"Парсинг: ОШИБКА (не удалось загрузить) — {target_url}", file=sys.stderr)
        sys.exit(2)

    soup = BeautifulSoup(response.text, "html.parser")
    diplodoc_fragment, diplodoc_data = extract_diplodoc_fragment(soup)
    diplodoc_blocks_text = collect_diplodoc_blocks_text(diplodoc_data or {})

    if diplodoc_fragment:
        text = extract_text(diplodoc_fragment)
        parse_mode = "diplodoc:data.html"
    elif diplodoc_blocks_text:
        text = diplodoc_blocks_text
        parse_mode = "diplodoc:data.data.blocks"
    else:
        main_container = choose_main_container(soup)
        text = extract_text(main_container)
        parse_mode = "html:main-container"

    title = extract_title(soup, diplodoc_data)

    print(f"Парсинг: OK ({parse_mode})")
    print(f'Заголовок: "{title}"')
    print(f"URL: {target_url}")
    print(f"Длина текста: {len(text)}")
    print(f"Первые {args.preview} символов: {text[: args.preview]}")
    print("==================================")

    hits = find_stopword_hits(text, stop_words)
    total_hits = sum(len(items) for items in hits.values())
    print(f"Всего стоп-слов найдено: {total_hits}")

    if not hits:
        print("Совпадений не найдено.")
    else:
        for key, items in sorted(hits.items(), key=lambda kv: len(kv[1]), reverse=True):
            label = key
            print(f'\nСлово/шаблон "{label}" ({len(items)} вхождений):')
            for hit in items:
                print(f"  - {hit.context}")

    # Диагностика: подстрока найдена, но по границам слова — нет.
    loose_found: list[str] = []
    for rule in stop_words:
        w = rule.raw.strip()
        if not w:
            continue
        if w.lower().startswith("re:"):
            continue
        try:
            if re.search(re.escape(w), text, re.IGNORECASE) and rule.key not in hits:
                loose_found.append(w)
        except re.error:
            continue

    if loose_found:
        print("\n[DEBUG] Найдено как подстрока, но не как целое слово (\\b...\\b):")
        for w in loose_found:
            print(f" - {w}")


if __name__ == "__main__":
    main()
