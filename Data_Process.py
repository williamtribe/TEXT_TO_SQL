import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()  # .env 불러오기

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXCEL_PATH = r"/mnt/c/Users/team42/projects_results/TEXT_TO_SQL/mafiaDB_RAG.xlsx"
OUTPUT_SHEET = "TABLE_SUMMARY"

# -------------------------------
# 네가 준 설명은 "시드(seed)" 설명
# -------------------------------
SEED_DESCRIPTIONS = {
    "EMOTICON": "인게임 내에서 사용되는 이모티콘의 목록과 그들의 item_id",
    "EVENT_PACKAGE_LOG": "유저가 한 이벤트 상자깡",
    "GAME_CHAT": "모든 유저가 모든 게임내에서 한 모든 채팅",
    "GAME_LOG": "게임의 기록 (게임ID와 메시지 남음)",
    "GAME_MEMBER": "유저들의 게임참여 기록",
    "GAME_PLAY": "게임의 기록 (게임ID와 게임시간, 게임결과, 게임의 종류)",
    "INAPP_PACKAGE": "인앱에서 구매할 수 있는 패키지의 종류",
    "INAPP_PACKAGE_ITEM": "INAPP_PACKAGE에서 살 수 있는 패키지들에 포함된 아이템들과 그 수량",
    "INVENTORY": "유저들의 인벤토리",
    "INVENTORY2": "유저들의 두번째 인벤토리",
    "INVENTORY_FAVORITE": "유저들의 인벤토리 중 즐겨찾기 아이템",
    "ITEM": "인게임에 존재하는 아이템들과 그들의 CODE, 이름",
    "ITEM_BUY_LOG": "유저들이 상점에서 인게임 재화를 이용해 아이템을 구매한 경우",
    "META_INAPP_PACKAGE": "인앱에서 구매할 수 있는 패키지의 종류 (개발자 전용)",
    "USER": "사용자별로 종합적인 정보 (닉네임, 최근 로그인 시간 등)",
    "USER_JOB_CARD_DECK": "유저들의 인게임 덱 슬롯 1",
    "USER_JOB_CARD_DECK2": "유저들의 인게임 덱 슬롯 2",
    "USER_JOB_CARD_DECK3": "유저들의 인게임 덱 슬롯 3",
    "USER_JOB_CARD_DECK4": "유저들의 인게임 덱 슬롯 4",
    "USER_JOB_CARD_DECK5": "유저들의 인게임 덱 슬롯 5",
    "USER_PHONE_VERIFICATION": "유저 계정별 인증 핸드폰 번호",
    "gacha_rate_items": "인게임 상자깡에서 뽑을 수 있는 아이템과 그 확률",
}

# -------------------------------
# GPT에게 final description 생성 요청
# -------------------------------
def generate_final_description(table_name, columns, seed_desc=None):
    col_str = ", ".join(columns)
    seed = seed_desc if seed_desc else "(설명 없음)"

    prompt = f"""
당신은 게임 서버(Mafia42) 데이터베이스 전문가입니다.

아래의 테이블 이름, 컬럼 목록, 그리고 기본 설명(seed description)을 기반으로
RAG(Text-to-SQL)에서 사용할 고품질 테이블 설명을 작성하세요.

조건:
- 2~3문장
- 테이블의 목적을 명확히 설명
- 주요 키(user_id, game_id, item_code 등) 관계도 설명
- JOIN 할 때 참고할 수 있는 단서를 포함
- SQL 쿼리를 만들 때 도움이 되는 방식으로 서술

[테이블 이름]
{table_name}

[컬럼 목록]
{col_str}

[기본 설명]
{seed}

[최종 설명]
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return response.output_text.strip()


# -------------------------------
# 엑셀 읽어서 TABLE_NAME 기준 그룹화
# -------------------------------
df = pd.read_excel(EXCEL_PATH)

tables = {}
for tbl, sub in df.groupby("TABLE_NAME"):
    tables[tbl] = sub["COLUMN_NAME"].tolist()

# -------------------------------
# Summary 생성
# -------------------------------
rows = []

for table_name, columns in tables.items():
    seed = SEED_DESCRIPTIONS.get(table_name)
    final_desc = generate_final_description(table_name, columns, seed)

    rows.append({
        "table_name": table_name,
        "columns": ", ".join(columns),
        "description": final_desc
    })

summary_df = pd.DataFrame(rows)

# -------------------------------
# 같은 엑셀 파일에 TABLE_SUMMARY 시트로 저장
# -------------------------------
with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    summary_df.to_excel(writer, sheet_name=OUTPUT_SHEET, index=False)

print("TABLE_SUMMARY 시트 생성 완료!")
