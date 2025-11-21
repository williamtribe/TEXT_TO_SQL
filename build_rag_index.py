# build_rag_index.py

import pandas as pd
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # .env 불러오기

EXCEL_PATH = r"/mnt/c/Users/team42/projects_results/TEXT_TO_SQL/mafiaDB_RAG.xlsx"
OUTPUT_JSON = r"/mnt/c/Users/team42/projects_results/TEXT_TO_SQL/rag_index.json"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_rag_index():
    df = pd.read_excel(EXCEL_PATH, sheet_name="TABLE_SUMMARY")

    rag_list = []

    for _, row in df.iterrows():
        table = row["table_name"]
        columns = row["columns"]
        desc = row["description"]

        text_block = f"""
Table: {table}
Columns: {columns}
Description: {desc}
""".strip()

        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=text_block
        ).data[0].embedding

        rag_list.append({
            "table_name": table,
            "text": text_block,
            "embedding": emb
        })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rag_list, f, ensure_ascii=False, indent=2)

    print("✅ rag_index.json 생성 완료:", OUTPUT_JSON)


if __name__ == "__main__":
    build_rag_index()
