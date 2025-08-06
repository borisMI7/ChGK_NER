import requests
import pandas as pd
import time
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chgk_parser.log"), logging.StreamHandler()],
)

BASE_URL = "http://www.db.chgk.info"


def get_args():
    parser = argparse.ArgumentParser(description="Скачивание вопросов с db.chgk.info.")
    parser.add_argument(
        "--output",
        type=str,
        default="chgk_questions.parquet",
        help="Путь к выходному Parquet-файлу.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить количество обрабатываемых пакетов (для отладки).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.2, help="Задержка между запросами в секундах."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Количество параллельных потоков для загрузки туров и вопросов.",
    )
    return parser.parse_args()


def fetch_data(
    session: requests.Session, url: str, delay: float
) -> Optional[Dict[str, Any]]:
    try:
        time.sleep(delay)
        response = session.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при запросе {url}: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Ошибка декодирования JSON с {url}: {e}")
    return None


def process_tour(
    tour_info: Dict[str, Any],
    package_info: Dict[str, Any],
    session: requests.Session,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    tour_questions = []
    tour_id = tour_info.get("id")
    if not tour_id:
        return []

    questions_url = f"{BASE_URL}/tours/{tour_id}/questions"
    questions_data = fetch_data(session, questions_url, args.delay)

    if questions_data and "hydra:member" in questions_data:
        for question_info in questions_data["hydra:member"]:
            flat_record = {
                "package_id": package_info.get("id"),
                "package_title": package_info.get("title"),
                "package_played_at": package_info.get("playedAt"),
                "package_editors": json.dumps(
                    package_info.get("editors"), ensure_ascii=False
                ),
                "package_info": package_info.get("info"),
                "tour_id": tour_id,
                "tour_title": tour_info.get("title"),
                "tour_number": tour_info.get("number"),
                "question_id": question_info.get("id"),
                "question_number": question_info.get("number"),
                "question_type": question_info.get("type"),
                "question_text": question_info.get("question"),
                "answer": question_info.get("answer"),
                "pass_criteria": question_info.get("passCriteria"),
                "authors": json.dumps(question_info.get("authors"), ensure_ascii=False),
                "sources": json.dumps(question_info.get("sources"), ensure_ascii=False),
                "comments": question_info.get("comments"),
            }
            tour_questions.append(flat_record)
    return tour_questions


def process_package(
    package_info: Dict[str, Any], session: requests.Session, args: argparse.Namespace
) -> List[Dict[str, Any]]:
    package_questions = []
    package_id = package_info.get("id")
    if not package_id:
        return []

    tours_url = f"{BASE_URL}/packages/{package_id}/tours"
    tours_data = fetch_data(session, tours_url, args.delay)

    if tours_data and "hydra:member" in tours_data:
        for tour_info in tours_data["hydra:member"]:
            package_questions.extend(
                process_tour(tour_info, package_info, session, args)
            )

    logging.info(
        f"Пакет ID {package_id} '{package_info.get('title')}': собрано {len(package_questions)} вопросов."
    )
    return package_questions


def main():
    args = get_args()
    all_questions_data = []

    output_file = Path(args.output)
    last_played_at = None

    if output_file.exists():
        logging.info(
            f"Найден существующий файл: {output_file}. Попытка возобновить загрузку."
        )
        try:
            df_existing = pd.read_parquet(output_file)
            if "package_played_at" in df_existing.columns and not df_existing.empty:
                df_existing["package_played_at"] = pd.to_datetime(
                    df_existing["package_played_at"], errors="coerce"
                )
                df_existing.dropna(subset=["package_played_at"], inplace=True)
                last_played_at = df_existing["package_played_at"].max().isoformat()
                logging.info(
                    f"Последняя дата в файле: {last_played_at}. Будут скачаны только более новые пакеты."
                )
        except Exception as e:
            logging.error(
                f"Не удалось прочитать существующий файл {output_file}: {e}. Начинаем с нуля."
            )

    with requests.Session() as session, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        current_page = 1
        packages_to_process = []

        logging.info("Фаза 1: Сбор списка пакетов...")
        while True:
            if args.limit is not None and len(packages_to_process) >= args.limit:
                logging.info(f"Достигнут лимит в {args.limit} пакетов для сбора.")
                break

            played_at_filter = (
                f"&playedAt[before]={last_played_at}" if last_played_at else ""
            )
            packages_url = f"{BASE_URL}/packages?page={current_page}&order[playedAt]=desc{played_at_filter}"

            packages_data = fetch_data(session, packages_url, args.delay)

            if (
                not packages_data
                or "hydra:member" not in packages_data
                or not packages_data["hydra:member"]
            ):
                logging.info("Больше пакетов не найдено.")
                break

            packages_to_process.extend(packages_data["hydra:member"])
            logging.info(
                f"Собрана страница {current_page}. Всего пакетов для обработки: {len(packages_to_process)}"
            )
            current_page += 1

        if not packages_to_process:
            logging.info("Нет новых пакетов для обработки.")
            return

        logging.info(
            f"Фаза 2: Обработка {len(packages_to_process)} пакетов с использованием {args.workers} потоков..."
        )

        future_to_package = {
            executor.submit(process_package, pkg, session, args): pkg
            for pkg in packages_to_process
        }

        for future in tqdm(
            as_completed(future_to_package),
            total=len(packages_to_process),
            desc="Обработка пакетов",
        ):
            try:
                questions = future.result()
                if questions:
                    all_questions_data.extend(questions)
            except Exception as e:
                package_title = future_to_package[future].get("title", "N/A")
                logging.error(f"Ошибка при обработке пакета '{package_title}': {e}")

    if not all_questions_data:
        logging.info("Не было собрано ни одного нового вопроса. Файл не будет изменен.")
        return

    logging.info(
        f"Сбор данных завершен. Всего собрано новых записей: {len(all_questions_data)}."
    )
    logging.info("Создание DataFrame...")

    df_new = pd.DataFrame(all_questions_data)

    if last_played_at:
        logging.info("Объединение новых данных с существующими...")
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.drop_duplicates(subset=["question_id"], keep="last", inplace=True)
    df_final.sort_values(by="package_played_at", ascending=False, inplace=True)

    try:
        logging.info(f"Сохранение DataFrame в Parquet файл: '{args.output}'...")
        df_final.to_parquet(args.output, engine="pyarrow", index=False)
        logging.info("Файл успешно сохранен.")
    except Exception as e:
        logging.error(f"Произошла ошибка при сохранении файла Parquet: {e}")

    logging.info("Работа скрипта завершена.")


if __name__ == "__main__":
    main()
