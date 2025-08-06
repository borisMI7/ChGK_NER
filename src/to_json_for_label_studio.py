import pandas as pd
import json
import random

parquet_file_path = "../data/got_q_db.parquet"
df = pd.read_parquet(parquet_file_path)


df["question"] = df["text"].fillna("")
df["answer"] = df["answer"].fillna("")
df["comment"] = df["comment"].fillna("")

tasks = []

for index, row in df.iterrows():
    question_part = row["question"]
    answer_part = f"\n\nОТВЕТ:\n{row['answer']}" if row["answer"] else ""
    comment_part = f"\n\nКОММЕНТАРИЙ:\n{row['comment']}" if row["comment"] else ""

    full_text = f"{question_part}{answer_part}{comment_part}"

    task = {"data": {"text": full_text.strip()}}
    tasks.append(task)

random.shuffle(tasks)
print("Список задач успешно перемешан.")

output_json_path = "../data/chgk_for_label_studio_combined_shuffled.json"

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(tasks, f, ensure_ascii=False, indent=4)

print(
    f"Данные успешно преобразованы, перемешаны и сохранены в файл: {output_json_path}"
)
