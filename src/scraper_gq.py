import requests
import requests.exceptions
from bs4 import BeautifulSoup
import tqdm
import json
import ast
import time
import random
import pandas as pd
import argparse
from pathlib import Path
from typing import Any, Optional, List, Dict
import logging


URL_BASE = "https://gotquestions.online"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
}
REQUEST_TIMEOUT = 20
RETRIES = 3
logger = logging.getLogger(__name__)


def safe_get(lst: List, index: int, default: Any = None) -> Any:
    return lst[index] if -len(lst) <= index < len(lst) else default


def extract_json_from_script(soup: BeautifulSoup) -> Optional[Dict]:
    try:
        scripts = soup.find_all("script")

        data = max(scripts, key=lambda x: len(x.get_text()), default=None)
        if not data:
            return None

        data = data.get_text()

        start_index = data.find('"') + 1
        end_index = data.rfind('"')
        data_string_with_prefix = data[start_index:end_index]

        json_string = data_string_with_prefix.lstrip("1234567890:")
        json_string = json_string.rstrip("\\n")

        unescaped_string = ast.literal_eval(f'"{json_string}"')

        return json.loads(unescaped_string)
    except (json.JSONDecodeError, SyntaxError) as e:
        logger.error(f"Ошибка декодирования JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при извлечении JSON: {e}")
        return None


def fetch_tournament_ids_from_page(
    session: requests.Session, page_num: int
) -> Optional[List[int]]:
    response = None
    for attempt in range(RETRIES):
        try:
            response = session.get(
                f"{URL_BASE}/?page={page_num}", timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            break
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Ошибка соединения при получении страницы {page_num}: {e}")
            if attempt < RETRIES - 1:
                delay = 2 ** (attempt + 1)
                logger.info(
                    f"Повторная попытка через {delay} сек (попытка {attempt + 2}/{RETRIES})..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Все {RETRIES} попыток для страницы {page_num} не удались. Пропускаем."
                )
                return None
        except requests.RequestException as e:
            logger.error(f"Ошибка сети при получении страницы {page_num}: {e}")
            return None
    try:
        soup = BeautifulSoup(response.text, "html.parser")

        parsed_data = extract_json_from_script(soup)
        if not parsed_data:
            return None

        pack_object = parsed_data[3]["children"][3]["packs"]
        ids = [pack["id"] for pack in pack_object]
        return ids
    except (KeyError, IndexError) as e:
        logger.warning(
            f"Структура JSON на странице {page_num} изменилась. Не удалось найти 'packs': {e}"
        )
        logger.debug(f"Содержимое страницы {page_num}:\n{soup.prettify()}")
        return None


def fetch_questions_from_pack(
    session: requests.Session, pack_id: int
) -> Optional[pd.DataFrame]:
    response = None
    for attempt in range(RETRIES):
        try:
            response = session.get(
                f"{URL_BASE}/pack/{pack_id}", timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            break
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Ошибка соединения при получении пакета {pack_id}: {e}")
            if attempt < RETRIES - 1:
                delay = 2 ** (attempt + 1)
                logger.info(
                    f"Повторная попытка через {delay} сек (попытка {attempt + 2}/{RETRIES})..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Все {RETRIES} попыток для пакета {pack_id} не удались. Пропускаем."
                )
                return None
        except requests.RequestException as e:
            logger.error(f"Ошибка сети при получении пакета {pack_id}: {e}")
            return None

    try:
        soup = BeautifulSoup(response.text, "html.parser")

        parsed_data = extract_json_from_script(soup)
        if not parsed_data:
            return None

        pack_object = parsed_data[3]["children"][3]["pack"]

        questions_list = []
        for tour in pack_object.get("tours", []):
            for q in tour.get("questions", []):
                processed_data = {
                    "id": q.get("id"),
                    "number": q.get("number"),
                    "text": q.get("text"),
                    "razdatkaText": q.get("razdatkaText"),
                    "razdatkaPic": q.get("razdatkaPic"),
                    "answer": q.get("answer"),
                    "zachet": q.get("zachet"),
                    "comment": q.get("comment"),
                    "source": q.get("source"),
                    "author": safe_get(
                        q.get("authors", []), 0, default={"name": None}
                    ).get("name"),
                    "title": q.get("packTitle"),
                    "date": q.get("endDate"),
                }
                questions_list.append(processed_data)

        if not questions_list:
            return None

        return pd.DataFrame(questions_list)

    except (KeyError, IndexError) as e:
        logger.warning(
            f"Структура JSON в пакете {pack_id} изменилась. Не удалось найти 'pack': {e}"
        )
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Сборщик вопросов с сайта gotquestions.online."
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=3,
        help="Количество страниц со списками турниров для сканирования.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gotquestions_data.parquet",
        help="Путь к выходному файлу в формате .parquet.",
    )
    parser.add_argument(
        "--min-delay",
        type=float,
        default=1.5,
        help="Минимальная задержка между запросами в секундах.",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=4.0,
        help="Максимальная задержка между запросами в секундах.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    log_file = output_path.with_suffix(".log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logger.info("Начинаем сбор данных...")

    session = requests.Session()
    session.headers.update(HEADERS)

    all_tournament_ids = []
    logger.info(f"Сканируем {args.pages} страниц для сбора ID турниров...")
    for page in tqdm.tqdm(range(1, args.pages + 1), desc="Сбор ID турниров"):
        time.sleep(random.uniform(args.min_delay, args.max_delay))
        ids = fetch_tournament_ids_from_page(session, page)
        if ids:
            all_tournament_ids.extend(ids)

    if not all_tournament_ids:
        logger.critical("Не удалось собрать ни одного ID турнира. Завершение работы.")
        return

    logger.info(f"Собрано {len(all_tournament_ids)} уникальных ID турниров.")

    all_dataframes = []
    logger.info("Начинаем сбор вопросов из каждого турнира...")
    for pack_id in tqdm.tqdm(all_tournament_ids, desc="Сбор вопросов"):
        time.sleep(random.uniform(args.min_delay, args.max_delay))
        df_pack = fetch_questions_from_pack(session, pack_id)
        if df_pack is not None and not df_pack.empty:
            all_dataframes.append(df_pack)

    if not all_dataframes:
        logger.warning(
            "Не удалось собрать данные ни по одному турниру. Завершение работы."
        )
        return

    logger.info("Объединяем все данные в один файл...")
    final_df = pd.concat(all_dataframes, ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        final_df.to_parquet(output_path, index=False)
        logger.info(f"SUCCESS: Данные успешно сохранены в файл: {output_path}")
        logger.info(f"Итого собрано вопросов: {len(final_df)}")
    except Exception as e:
        logger.error(f"Не удалось сохранить файл. Ошибка: {e}")


if __name__ == "__main__":
    main()
