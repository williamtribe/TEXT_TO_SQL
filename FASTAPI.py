# FASTAPI.py
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
import os
import pymysql
from dotenv import load_dotenv
import httpx
from openai import OpenAI
import json
import numpy as np
import gspread
from gspread.exceptions import WorksheetNotFound
from google.oauth2.service_account import Credentials
from datetime import datetime
import logging
import random
import asyncio
import threading
import time
from typing import Optional

load_dotenv()  # .env 불러오기

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sql_query.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT"))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DB = os.getenv("MYSQL_DB")
RAG_INDEX_PATH = r"/mnt/c/Users/team42/projects_results/TEXT_TO_SQL/rag_index.json"
GOOGLE_SHEETS_CREDENTIALS_PATH = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "")
GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID", "")
GOOGLE_SHEETS_WORKSHEET_NAME = os.getenv("GOOGLE_SHEETS_WORKSHEET_NAME", "TABLE_SUMMARY")
GOOGLE_SHEETS_FEEDBACK_WORKSHEET_NAME = os.getenv("GOOGLE_SHEETS_FEEDBACK_WORKSHEET_NAME", "FEEDBACK")
GOOGLE_SHEETS_LOG_WORKSHEET_NAME = os.getenv("GOOGLE_SHEETS_LOG_WORKSHEET_NAME", "LOG")

with open(RAG_INDEX_PATH, "r", encoding="utf-8") as f:
    RAG_INDEX = json.load(f)

# 구글 시트 클라이언트 초기화 (선택적)
google_sheets_client = None
if GOOGLE_SHEETS_CREDENTIALS_PATH and os.path.exists(GOOGLE_SHEETS_CREDENTIALS_PATH):
    try:
        # 읽기/쓰기 권한 필요 (피드백 저장을 위해)
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/spreadsheets'
        ]
        creds = Credentials.from_service_account_file(GOOGLE_SHEETS_CREDENTIALS_PATH, scopes=scopes)
        google_sheets_client = gspread.authorize(creds)
        logger.info("✅ 구글 시트 클라이언트 초기화 완료")
        
        # 피드백 워크시트가 없으면 생성
        try:
            spreadsheet = google_sheets_client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
            try:
                spreadsheet.worksheet(GOOGLE_SHEETS_FEEDBACK_WORKSHEET_NAME)
            except WorksheetNotFound:
                # 피드백 워크시트가 없으면 생성
                worksheet = spreadsheet.add_worksheet(
                    title=GOOGLE_SHEETS_FEEDBACK_WORKSHEET_NAME,
                    rows=1000,
                    cols=10
                )
                # 헤더 추가
                worksheet.append_row([
                    "타임스탬프", "사용자명", "피드백", "질문", "SQL", "결과", "메시지ID", "사용자ID"
                ])
                logger.info(f"✅ 피드백 워크시트 생성 완료: {GOOGLE_SHEETS_FEEDBACK_WORKSHEET_NAME}")
            
            # LOG 워크시트가 없으면 생성
            try:
                spreadsheet.worksheet(GOOGLE_SHEETS_LOG_WORKSHEET_NAME)
            except WorksheetNotFound:
                # LOG 워크시트가 없으면 생성
                worksheet = spreadsheet.add_worksheet(
                    title=GOOGLE_SHEETS_LOG_WORKSHEET_NAME,
                    rows=10000,
                    cols=10
                )
                # 헤더 추가
                worksheet.append_row([
                    "타임스탬프", "사용자명", "질문", "SQL", "결과", "피드백", "메시지ID", "사용자ID"
                ])
                logger.info(f"✅ LOG 워크시트 생성 완료: {GOOGLE_SHEETS_LOG_WORKSHEET_NAME}")
        except Exception as e:
            logger.warning(f"⚠️ 워크시트 확인/생성 실패: {e}")
    except Exception as e:
        logger.warning(f"⚠️ 구글 시트 클라이언트 초기화 실패: {e}")

def cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def rag_retrieve(question: str, top_k: int = 3):
    # 질문 임베딩
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    scored = []
    for item in RAG_INDEX:
        score = cosine(q_emb, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [s[1] for s in scored[:top_k]]
    
    return results


# ------------------- 구글 시트에서 스키마 정보 가져오기 -------------------
def get_schema_from_google_sheets() -> str:
    """구글 시트에서 전체 스키마 정보를 가져와서 텍스트로 반환"""
    if not google_sheets_client or not GOOGLE_SHEETS_SPREADSHEET_ID:
        return None
    
    try:
        spreadsheet = google_sheets_client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
        worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_WORKSHEET_NAME)
        
        # 모든 데이터 가져오기
        records = worksheet.get_all_records()
        
        schema_blocks = []
        for record in records:
            table = record.get("table_name", "")
            columns = record.get("columns", "")
            desc = record.get("description", "")
            
            schema_blocks.append(f"""
Table: {table}
Columns: {columns}
Description: {desc}
""".strip())
        
        return "\n\n---\n\n".join(schema_blocks)
    except Exception as e:
        print(f"⚠️ 구글 시트에서 스키마 가져오기 실패: {e}")
        return None


client = OpenAI(api_key=OPENAI_API_KEY)


# ------------------- ① Slack Event 엔드포인트 -------------------
@app.post("/slack/events")
async def slack_events(request: Request):
    try:
        body = await request.json()
        logger.info(f"📥 Slack Event 수신: {json.dumps(body, ensure_ascii=False, indent=2)}")

        # URL 검증 (challenge 응답)
        if "challenge" in body:
            challenge_value = body["challenge"]
            logger.info(f"✅ Challenge 요청 수신: {challenge_value}")
            return JSONResponse(content={"challenge": challenge_value})

        # 이벤트 타입 확인
        event = body.get("event", {})
        event_type = event.get("type", "")
        
        # App Home 열림 이벤트 처리
        if event_type == "app_home_opened":
            user_id = event.get("user", "")
            logger.info(f"🏠 App Home 열림 (사용자: {user_id})")
            
            # App Home 업데이트 (백그라운드에서 처리)
            asyncio.create_task(update_app_home(user_id))
            
            return JSONResponse(content={"ok": True})
    except Exception as e:
        logger.error(f"❌ Slack Event 파싱 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # challenge 요청이어도 에러가 나면 200 응답
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=200)

    # 이벤트 타입 확인 (이미 위에서 확인했지만, 다른 이벤트 처리용)
    event = body.get("event", {})
    event_type = event.get("type", "")
    text = event.get("text", "")
    channel = event.get("channel", "")
    user = event.get("user", "")

    # DM 메시지 이벤트 처리 (message.im)
    if event_type == "message" and channel.startswith("D"):
        logger.info("=" * 80)
        logger.info(f"💬 DM 메시지 수신 (사용자: {user}, 채널: {channel}): {text}")
        
        # 봇 메시지는 무시
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            logger.info("🤖 봇 메시지 무시")
            return JSONResponse(content={"ok": True})
        
        # DM에서는 슬래시 커맨드 없이 일반 메시지로 처리
        if text and text.strip():
            query_text = text.strip()
            # 슬래시로 시작하면 제거 (예: /sql 질문 -> 질문)
            if query_text.startswith("/"):
                # /sql 또는 /로 시작하는 경우 슬래시 부분 제거
                parts = query_text.split(None, 1)
                if len(parts) > 1:
                    query_text = parts[1]
                else:
                    # 슬래시만 있으면 무시
                    return JSONResponse(content={"ok": True})
            
            logger.info(f"🚀 DM에서 쿼리 요청: {query_text}")
            
            # 사용자 정보 가져오기 (user_name을 위해)
            user_name = "사용자"  # 기본값
            try:
                user_info_response = httpx.post(
                    "https://slack.com/api/users.info",
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    json={"user": user}
                )
                user_info = user_info_response.json()
                if user_info.get("ok"):
                    user_name = user_info.get("user", {}).get("name", "사용자")
            except Exception as e:
                logger.warning(f"⚠️ 사용자 정보 가져오기 실패: {e}")
            
            # 즉시 응답 메시지 전송 (슬래시 커맨드와 동일하게)
            start_messages = ["체크해 봐야겠군.", "움직일 시간인가."]
            start_message = random.choice(start_messages)
            
            try:
                logger.info(f"📤 초기 응답 메시지 전송: {start_message}")
                response = httpx.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    json={
                        "channel": channel,
                        "text": start_message
                    }
                )
                response_data = response.json()
                if response_data.get("ok"):
                    logger.info(f"✅ 초기 응답 메시지 전송 성공")
                else:
                    logger.error(f"❌ 초기 응답 메시지 전송 실패: {response_data.get('error')}")
            except Exception as e:
                logger.error(f"❌ 초기 응답 메시지 전송 예외: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # 백그라운드에서 처리 (user_name 전달)
            query_id = f"{user}_{datetime.now().timestamp()}"
            # DM 채널 ID는 이미 channel에 있음
            asyncio.create_task(process_query_async(query_text, "", user_name, user, query_id, channel))
        
        logger.info("=" * 80)
        return JSONResponse(content={"ok": True})

    # 채널 메시지 처리 (슬래시 커맨드 또는 "쿼리" 키워드)
    if text.startswith("/sql") or "쿼리" in text:
        query_text = text.replace("/sql", "").strip()
        logger.info("=" * 80)
        logger.info(f"🚀 새로운 쿼리 요청: {query_text}")
        
        sql = generate_sql_with_gpt(query_text, use_full_schema=False)
        result = execute_sql(sql)
        
        # 에러 발생 시 구글 시트에서 전체 스키마로 재시도
        if result.startswith("SQL 실행 오류") or result.startswith("오류 발생"):
            logger.warning(f"⚠️ 첫 번째 시도 실패. 구글 시트에서 전체 스키마로 재시도...")
            sql = generate_sql_with_gpt(query_text, use_full_schema=True)
            result = execute_sql(sql)
            
            if result.startswith("SQL 실행 오류") or result.startswith("오류 발생"):
                error_msg = f"❌ 쿼리 실행 중 오류가 발생했습니다.\n\n```sql\n{sql}\n```\n\n오류: {result}\n\n구글 시트의 스키마 정보를 확인해주세요."
                send_message_with_feedback(channel, error_msg, query_text, sql, result)
            else:
                success_msg = f"✅ 재시도 성공!\n\n```sql\n{sql}\n```\n\n결과:\n{result}"
                send_message_with_feedback(channel, success_msg, query_text, sql, result)
        else:
            normal_msg = f"```sql\n{sql}\n```\n\n결과:\n{result}"
            send_message_with_feedback(channel, normal_msg, query_text, sql, result)
        
        logger.info("=" * 80)

    return JSONResponse(content={"ok": True})


# ------------------- ③ Slack Interactive Actions (피드백 버튼) 엔드포인트 -------------------
@app.post("/slack/interactivity")
async def slack_interactivity(request: Request, background_tasks: BackgroundTasks):
    """Slack 버튼 클릭 시 피드백을 받는 엔드포인트"""
    try:
        # Slack Interactive Components는 form-data로 payload를 보냄
        form = await request.form()
        payload_str = form.get("payload", "{}")
        
        # JSON 문자열인 경우 파싱
        if isinstance(payload_str, str):
            payload = json.loads(payload_str)
        else:
            payload = payload_str
    except Exception as e:
        # form-data가 아닌 경우 JSON body로 시도
        try:
            body = await request.json()
            payload = body
        except:
            logger.error(f"❌ 피드백 파싱 실패: {e}")
            return JSONResponse(content={"ok": False, "error": "Invalid payload"}, status_code=400)
    
    logger.info("=" * 80)
    logger.info(f"📥 피드백 수신: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    # actions 확인
    actions = payload.get("actions", [])
    if not actions:
        logger.warning("⚠️ actions가 없습니다. payload 구조를 확인하세요.")
        return JSONResponse(content={"ok": False, "error": "No actions found"}, status_code=400)
    
    action = actions[0]
    action_value = action.get("value", "{}")
    
    try:
        if isinstance(action_value, str):
            value_data = json.loads(action_value)
        else:
            value_data = action_value
    except json.JSONDecodeError as e:
        logger.error(f"❌ value JSON 파싱 실패: {e}, value: {action_value}")
        return JSONResponse(content={"ok": False, "error": "Invalid value format"}, status_code=400)
    
    # 취소 액션 처리
    action_type = value_data.get("action", "")
    if action_type == "cancel":
        query_id = value_data.get("query_id", "")
        if query_id and query_id in running_queries:
            running_queries[query_id]["cancelled"] = True
            logger.info(f"🚫 쿼리 취소 요청: {query_id}")
        
        response_url = payload.get("response_url", "")
        if response_url:
            httpx.post(
                response_url,
                json={
                    "text": "❌ 취소 요청 처리 중...",
                    "replace_original": False
                }
            )
        return JSONResponse(content={"ok": True})
    
    feedback_type = value_data.get("feedback", "unknown")
    question = value_data.get("question", "")
    sql = value_data.get("sql", "")
    result = value_data.get("result", "")
    message_id = value_data.get("message_id", "")
    user = payload.get("user", {})
    user_name = user.get("name", "unknown")
    response_url = payload.get("response_url", "")
    
    # 즉시 Slack에 응답 (3초 이내)
    if response_url:
        if feedback_type == "positive":
            # 긍정적 피드백 메시지 (랜덤)
            success_messages = ["목표 대상 처리 완료.", "임무를 마쳤다."]
            message = random.choice(success_messages)
        else:
            # 부정적 피드백 메시지 (랜덤)
            negative_messages = ["칫, 방심했군...!", "윽… 꼬리를 밟히다니."]
            message = random.choice(negative_messages)
        
        httpx.post(
            response_url,
            json={
                "text": message,
                "replace_original": False
            }
        )
    
    # 백그라운드에서 피드백 저장 (구글 시트 작업은 느릴 수 있음)
    background_tasks.add_task(
        save_feedback_background,
        feedback_type, question, sql, result, message_id, user_name, user.get("id", "")
    )
    
    return JSONResponse(content={"ok": True})


# ------------------- 백그라운드 피드백 저장 함수 -------------------
def save_feedback_background(feedback_type: str, question: str, sql: str, result: str, message_id: str, user_name: str, user_id: str):
    """백그라운드에서 피드백 저장 (구글 시트 작업)"""
    try:
        timestamp = datetime.now().isoformat()
        feedback_log = {
            "timestamp": timestamp,
            "message_id": message_id,
            "user_id": user_id,
            "user_name": user_name,
            "feedback": feedback_type,
            "question": question,
            "sql": sql,
            "result": result[:500] if result else ""
        }
        
        # 1) 파일에 저장
        feedback_file = "feedback_log.jsonl"
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_log, ensure_ascii=False) + "\n")
        
        # 2) 구글 시트 FEEDBACK 워크시트에 저장 (피드백 버튼 클릭 시)
        if google_sheets_client and GOOGLE_SHEETS_SPREADSHEET_ID:
            try:
                spreadsheet = google_sheets_client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
                worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_FEEDBACK_WORKSHEET_NAME)
                
                # 행 추가
                row = [
                    timestamp,
                    user_name,
                    feedback_type,
                    question,
                    sql,
                    result[:500] if result else "",
                    message_id,
                    user_id
                ]
                worksheet.append_row(row)
                logger.info(f"✅ 구글 시트 FEEDBACK에 피드백 저장 완료")
            except Exception as e:
                logger.error(f"❌ 구글 시트 FEEDBACK에 피드백 저장 실패: {e}")
        
        # 3) 구글 시트 LOG 워크시트의 해당 행 업데이트 (피드백 추가)
        if google_sheets_client and GOOGLE_SHEETS_SPREADSHEET_ID and message_id:
            try:
                spreadsheet = google_sheets_client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
                worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_LOG_WORKSHEET_NAME)
                
                # 메시지 ID로 해당 행 찾기
                all_values = worksheet.get_all_values()
                for idx, row in enumerate(all_values[1:], start=2):  # 헤더 제외, 2행부터 시작
                    if len(row) > 6 and row[6] == message_id:  # 메시지ID 컬럼 (7번째, 인덱스 6)
                        # 피드백 컬럼 업데이트 (6번째 컬럼, 인덱스 5)
                        worksheet.update_cell(idx, 6, feedback_type)  # 피드백 컬럼 업데이트
                        logger.info(f"✅ 구글 시트 LOG의 피드백 업데이트 완료 (행 {idx})")
                        break
            except Exception as e:
                logger.error(f"❌ 구글 시트 LOG 피드백 업데이트 실패: {e}")
        
        logger.info(f"💾 피드백 저장 완료: {feedback_type} (사용자: {user_name})")
        logger.info(f"   질문: {question}")
        logger.info(f"   SQL: {sql}")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"❌ 피드백 저장 백그라운드 작업 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())


# ------------------- 구글 시트에 쿼리 로그 저장 -------------------
def save_query_to_google_sheets(user_name: str, user_id: str, question: str, sql: str, result: str, message_id: str, feedback: str = ""):
    """구글 시트 LOG 워크시트에 쿼리 실행 기록 저장 (FEEDBACK과 동일한 방식)"""
    if not google_sheets_client or not GOOGLE_SHEETS_SPREADSHEET_ID:
        return
    
    # 락을 사용하여 순차적으로 저장
    with google_sheets_lock:
        try:
            spreadsheet = google_sheets_client.open_by_key(GOOGLE_SHEETS_SPREADSHEET_ID)
            worksheet = spreadsheet.worksheet(GOOGLE_SHEETS_LOG_WORKSHEET_NAME)
            
            timestamp = datetime.now().isoformat()
            row = [
                timestamp,
                user_name,
                question,
                sql,
                result[:1000] if result else "",  # 결과는 최대 1000자
                feedback,  # 피드백이 없으면 빈 문자열
                message_id,
                user_id
            ]
            worksheet.append_row(row)
            logger.info(f"✅ 구글 시트 LOG에 쿼리 기록 저장 완료")
        except Exception as e:
            logger.error(f"❌ 구글 시트 LOG에 쿼리 기록 저장 실패: {e}")


# 전역 변수: 실행 중인 쿼리 추적 (취소용)
running_queries = {}

# 구글 시트 쓰기 락 (순차적 저장을 위해)
google_sheets_lock = threading.Lock()


# ------------------- App Home 업데이트 함수 -------------------
async def update_app_home(user_id: str):
    """App Home 뷰 업데이트"""
    try:
        home_blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*통계업자*에 오신 것을 환영합니다.\n\n이 봇은 자연어 질문을 SQL 쿼리로 변환하여 데이터베이스에서 정보를 조회합니다."
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*사용 방법*\n\n• DM에서 질문하기: 봇과의 DM에서 자연어로 질문하세요.\n• 슬래시 커맨드: `/sql 질문` 형식으로 사용하세요."
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*예시*\n• `2018년 10월 29일부터 11월 30일까지 이벤트 게임 참여한 유저 카운팅`\n• `/sql 유저 수 세줘`"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "체크해 봐야겠군. | 움직일 시간인가."
                    }
                ]
            }
        ]
        
        logger.info(f"📤 views.publish 호출: user_id={user_id}")
        response = httpx.post(
            "https://slack.com/api/views.publish",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={
                "user_id": user_id,
                "view": {
                    "type": "home",
                    "blocks": home_blocks
                }
            }
        )
        response_data = response.json()
        logger.info(f"📥 views.publish 응답: {response_data}")
        
        if response_data.get("ok"):
            logger.info(f"✅ App Home 업데이트 완료 (사용자: {user_id})")
        else:
            error = response_data.get("error", "Unknown error")
            logger.error(f"❌ App Home 업데이트 실패: {error}")
            logger.error(f"   전체 응답: {response_data}")
            if "needed" in response_data:
                logger.error(f"   필요한 권한: {response_data.get('needed')}")
    except Exception as e:
        logger.error(f"❌ App Home 업데이트 예외: {e}")
        import traceback
        logger.error(traceback.format_exc())

# ------------------- 쿼리 처리 함수 (백그라운드) -------------------
async def process_query_async(text: str, response_url: str, user_name: str, user_id: str = "", query_id: str = "", channel_id: str = ""):
    """비동기로 쿼리를 처리하고 Slack에 결과 전송"""
    logger.info(f"🚀 쿼리 처리 시작 (사용자: {user_name}): {text}")

    # 1) GPT로 SQL 생성 (먼저 RAG 사용)
    sql = generate_sql_with_gpt(text, use_full_schema=False)
    
    # SQL 생성 후 즉시 메시지 전송 (진행 중 표시)
    message_id = f"{datetime.now().timestamp()}_{hash(text + sql)}"
    initial_message = f"질문: {text}\n\n```sql\n{sql}\n```\n\n⏳ 쿼리 실행 중..."
    
    initial_blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": initial_message
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "❌ 취소"
                    },
                    "style": "danger",
                    "value": json.dumps({"action": "cancel", "query_id": query_id})
                }
            ]
        }
    ]
    
    # 초기 메시지 전송 (chat.postMessage 사용하여 ts와 channel 얻기)
    message_ts = None
    logger.info(f"📤 메시지 전송 시도 (channel_id: {channel_id}, user_id: {user_id})")
    # channel_id가 있으면 chat.postMessage로 직접 전송
    if channel_id:
        try:
            logger.info(f"📤 chat.postMessage 호출: channel={channel_id}")
            initial_response = httpx.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                json={
                    "channel": channel_id,
                    "text": initial_message,
                    "blocks": initial_blocks
                }
            )
            initial_data = initial_response.json()
            logger.info(f"📥 chat.postMessage 응답: {initial_data}")
            if initial_data.get("ok"):
                message_ts = initial_data.get("ts")
                channel_id = initial_data.get("channel")
                logger.info(f"✅ 메시지 전송 성공: ts={message_ts}, channel={channel_id}")
            else:
                error_msg = initial_data.get('error', 'Unknown error')
                logger.error(f"❌ 메시지 전송 실패: {error_msg}")
                logger.error(f"   응답 데이터: {initial_data}")
                # 실패 시 response_url로 폴백
                if response_url:
                    httpx.post(
                        response_url,
                        json={
                            "response_type": "in_channel",
                            "text": initial_message,
                            "blocks": initial_blocks
                        }
                    )
        except Exception as e:
            logger.error(f"❌ chat.postMessage 실패: {e}")
            # 예외 발생 시 response_url로 폴백
            if response_url:
                httpx.post(
                    response_url,
                    json={
                        "response_type": "in_channel",
                        "text": initial_message,
                        "blocks": initial_blocks
                    }
                )
    else:
        # channel_id가 없으면 DM 채널 열기 시도 후 다시 시도
        logger.warning("⚠️ channel_id가 없음. DM 채널 열기 시도...")
        if user_id:
            dm_channel = open_dm_channel(user_id)
            if dm_channel:
                channel_id = dm_channel
                logger.info(f"✅ DM 채널 열기 성공, 메시지 재전송 시도: {channel_id}")
                try:
                    initial_response = httpx.post(
                        "https://slack.com/api/chat.postMessage",
                        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                        json={
                            "channel": channel_id,
                            "text": initial_message,
                            "blocks": initial_blocks
                        }
                    )
                    initial_data = initial_response.json()
                    if initial_data.get("ok"):
                        message_ts = initial_data.get("ts")
                        channel_id = initial_data.get("channel")
                        logger.info(f"✅ 메시지 전송 성공: ts={message_ts}, channel={channel_id}")
                    else:
                        error_msg = initial_data.get('error', 'Unknown error')
                        logger.error(f"❌ 메시지 전송 실패: {error_msg}")
                        logger.error(f"   응답 데이터: {initial_data}")
                        # 실패 시 response_url로 폴백
                        if response_url:
                            httpx.post(
                                response_url,
                                json={
                                    "response_type": "in_channel",
                                    "text": initial_message,
                                    "blocks": initial_blocks
                                }
                            )
                except Exception as e:
                    logger.error(f"❌ chat.postMessage 재시도 실패: {e}")
                    # 예외 발생 시 response_url로 폴백
                    if response_url:
                        httpx.post(
                            response_url,
                            json={
                                "response_type": "in_channel",
                                "text": initial_message,
                                "blocks": initial_blocks
                            }
                        )
            else:
                logger.error("❌ DM 채널 열기 실패")
                # response_url로 폴백
                if response_url:
                    httpx.post(
                        response_url,
                        json={
                            "response_type": "in_channel",
                            "text": initial_message,
                            "blocks": initial_blocks
                        }
                    )
        else:
            # user_id도 없으면 response_url 사용
            logger.warning("⚠️ user_id도 없어 response_url 사용")
            if response_url:
                httpx.post(
                    response_url,
                    json={
                        "response_type": "in_channel",
                        "text": initial_message,
                        "blocks": initial_blocks
                    }
                )
    
    # 쿼리 실행 시작 시간
    start_time = time.time()
    running_queries[query_id] = {"cancelled": False, "start_time": start_time, "message_ts": message_ts, "channel_id": channel_id}
    
    # 2) SQL 실행 (별도 스레드에서 실행하여 취소 가능하게)
    def run_sql():
        if query_id in running_queries and running_queries[query_id]["cancelled"]:
            return "취소됨"
        return execute_sql(sql)
    
    result = await asyncio.to_thread(run_sql)
    
    # 쿼리 완료 처리
    if query_id in running_queries:
        if running_queries[query_id]["cancelled"]:
            cancelled_message = f"질문: {text}\n\n```sql\n{sql}\n```\n\n❌ 쿼리가 취소되었습니다."
            # 메시지 업데이트 시도 (타임스탬프가 있으면)
            query_info = running_queries[query_id]
            if query_info.get("message_ts") and query_info.get("channel_id"):
                try:
                    httpx.post(
                        "https://slack.com/api/chat.update",
                        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                        json={
                            "channel": query_info["channel_id"],
                            "ts": query_info["message_ts"],
                            "text": cancelled_message
                        }
                    )
                except:
                    # 업데이트 실패 시 새 메시지로 전송
                    if response_url:
                        httpx.post(response_url, json={"text": cancelled_message})
            else:
                if response_url:
                    httpx.post(response_url, json={"text": cancelled_message})
            del running_queries[query_id]
            return
        del running_queries[query_id]
    
    # 메시지 ID 생성 (피드백 버튼과 로그 연결용)
    message_id = f"{datetime.now().timestamp()}_{hash(text + sql)}"

    # 3) 에러 발생 시 구글 시트에서 전체 스키마로 재시도
    if result.startswith("SQL 실행 오류") or result.startswith("오류 발생"):
        logger.warning(f"⚠️ 첫 번째 시도 실패. 구글 시트에서 전체 스키마로 재시도...")
        
        # 재시도 진행 상황 표시 (메시지 업데이트)
        retry_message = f"질문: {text}\n\n```sql\n{sql}\n```\n\n⚠️ 첫 번째 시도 실패. 재시도 중... (구글 시트 스키마 사용)"
        if query_id in running_queries:
            query_info = running_queries[query_id]
            if query_info.get("message_ts") and query_info.get("channel_id"):
                try:
                    httpx.post(
                        "https://slack.com/api/chat.update",
                        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                        json={
                            "channel": query_info["channel_id"],
                            "ts": query_info["message_ts"],
                            "text": retry_message
                        }
                    )
                except:
                    pass
        
        sql = generate_sql_with_gpt(text, use_full_schema=True)
        result = await asyncio.to_thread(execute_sql, sql)
        message_id = f"{datetime.now().timestamp()}_{hash(text + sql)}"  # 재시도 시 메시지 ID 재생성
        
        # 재시도 후에도 에러면 에러 메시지 포함해서 전송
        if result.startswith("SQL 실행 오류") or result.startswith("오류 발생"):
            error_message = f"질문: {text}\n\n```sql\n{sql}\n```\n\n결과:\n{result}\n\n⚠️ 오류 발생: {result}\n\n구글 시트의 스키마 정보를 확인해주세요."
        else:
            error_message = f"질문: {text}\n\n```sql\n{sql}\n```\n\n결과:\n{result}"
        
        # 구글 시트에 로그 저장 (피드백 없음)
        save_query_to_google_sheets(user_name, user_id, text, sql, result, message_id, feedback="")
        
        # 피드백 버튼 포함 메시지
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": error_message
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "👍 좋아요"
                        },
                        "style": "primary",
                        "value": json.dumps({"feedback": "positive", "question": text, "sql": sql, "result": result[:500], "message_id": message_id})
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "아쉬워요 ㅠㅠ"
                        },
                        "style": "danger",
                        "value": json.dumps({"feedback": "negative", "question": text, "sql": sql, "result": result[:500], "message_id": message_id})
                    }
                ]
            }
        ]
        
        # 메시지 업데이트 (타임스탬프가 있으면)
        query_info = running_queries.get(query_id, {}) if query_id in running_queries else {}
        if query_info.get("message_ts") and query_info.get("channel_id"):
            try:
                update_response = httpx.post(
                    "https://slack.com/api/chat.update",
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    json={
                        "channel": query_info["channel_id"],
                        "ts": query_info["message_ts"],
                        "text": error_message,
                        "blocks": blocks
                    }
                )
                update_data = update_response.json()
                if not update_data.get("ok"):
                    logger.error(f"❌ 메시지 업데이트 실패: {update_data.get('error')}")
                    # 업데이트 실패 시 새 메시지로 전송
                    if response_url:
                        httpx.post(response_url, json={"text": error_message, "blocks": blocks})
            except Exception as e:
                logger.error(f"❌ 메시지 업데이트 예외: {e}")
                # 업데이트 실패 시 새 메시지로 전송
                if response_url:
                    httpx.post(response_url, json={"text": error_message, "blocks": blocks})
        else:
            # 타임스탬프가 없으면 새 메시지로 전송
            if response_url:
                httpx.post(response_url, json={"text": error_message, "blocks": blocks})
    else:
        # 정상 실행 - 구글 시트에 로그 저장 (피드백 없음)
        save_query_to_google_sheets(user_name, user_id, text, sql, result, message_id, feedback="")
        
        # 정상 실행 - 질문 포함
        normal_message = f"질문: {text}\n\n```sql\n{sql}\n```\n\n결과:\n{result}"
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": normal_message
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "👍 좋아요"
                        },
                        "style": "primary",
                        "value": json.dumps({"feedback": "positive", "question": text, "sql": sql, "result": result[:500], "message_id": message_id})
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "아쉬워요 ㅠㅠ"
                        },
                        "style": "danger",
                        "value": json.dumps({"feedback": "negative", "question": text, "sql": sql, "result": result[:500], "message_id": message_id})
                    }
                ]
            }
        ]
        
        # 메시지 업데이트 (타임스탬프가 있으면)
        query_info = running_queries.get(query_id, {}) if query_id in running_queries else {}
        if query_info.get("message_ts") and query_info.get("channel_id"):
            try:
                update_response = httpx.post(
                    "https://slack.com/api/chat.update",
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    json={
                        "channel": query_info["channel_id"],
                        "ts": query_info["message_ts"],
                        "text": normal_message,
                        "blocks": blocks
                    }
                )
                update_data = update_response.json()
                if not update_data.get("ok"):
                    logger.error(f"❌ 메시지 업데이트 실패: {update_data.get('error')}")
                    # 업데이트 실패 시 새 메시지로 전송
                    if response_url:
                        httpx.post(response_url, json={"text": normal_message, "blocks": blocks})
                    elif query_info.get("channel_id"):
                        # DM인 경우 chat.postMessage로 새 메시지 전송
                        httpx.post(
                            "https://slack.com/api/chat.postMessage",
                            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                            json={
                                "channel": query_info["channel_id"],
                                "text": normal_message,
                                "blocks": blocks
                            }
                        )
            except Exception as e:
                logger.error(f"❌ 메시지 업데이트 예외: {e}")
                # 업데이트 실패 시 새 메시지로 전송
                if response_url:
                    httpx.post(response_url, json={"text": normal_message, "blocks": blocks})
                elif query_info.get("channel_id"):
                    # DM인 경우 chat.postMessage로 새 메시지 전송
                    httpx.post(
                        "https://slack.com/api/chat.postMessage",
                        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                        json={
                            "channel": query_info["channel_id"],
                            "text": normal_message,
                            "blocks": blocks
                        }
                    )
        else:
            # 타임스탬프가 없으면 새 메시지로 전송
            if response_url:
                httpx.post(response_url, json={"text": normal_message, "blocks": blocks})
            elif channel_id:
                # DM인 경우 chat.postMessage로 새 메시지 전송
                logger.info(f"📤 DM에 최종 결과 전송: channel={channel_id}")
                httpx.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    json={
                        "channel": channel_id,
                        "text": normal_message,
                        "blocks": blocks
                    }
                )


# ------------------- ② Slash Command 엔드포인트 -------------------
@app.post("/slack/command")
async def slack_command(request: Request, background_tasks: BackgroundTasks):
    form = await request.form()
    logger.info(f"📥 Slash Command 수신 (사용자: {form.get('user_name')}): {form.get('text')}")
    logger.info(f"📋 Form 데이터: {dict(form)}")  # 디버깅용

    text = form.get("text")
    response_url = form.get("response_url")
    user_name = form.get("user_name")
    user_id = form.get("user_id", "")
    channel_id = form.get("channel_id", "")
    channel_name = form.get("channel_name", "")

    logger.info(f"📋 슬래시 커맨드 데이터: user_id={user_id}, channel_id={channel_id}, channel_name={channel_name}")
    
    # DM인지 확인 (channel_name이 "directmessage"이거나 channel_id가 D로 시작하면 DM)
    is_dm = (channel_name == "directmessage" or (channel_id and channel_id.startswith("D"))) if channel_id else True
    
    # DM이거나 channel_id가 없으면 conversations.open으로 DM 채널 열기
    if is_dm or not channel_id:
        logger.info(f"🔄 DM 채널 열기 시도 (user_id: {user_id}, channel_id: {channel_id}, channel_name: {channel_name})")
        dm_channel = open_dm_channel(user_id)
        if dm_channel:
            channel_id = dm_channel
            logger.info(f"✅ DM 채널 열기 성공: {channel_id}")
        else:
            logger.error(f"❌ DM 채널 열기 실패. user_id: {user_id}")
            # 실패해도 원래 channel_id가 있으면 사용 시도
            if not channel_id:
                logger.error("❌ channel_id도 없어 메시지 전송 불가")
    else:
        logger.info(f"✅ 채널 ID 확인: {channel_id} (채널명: {channel_name})")

    # 쿼리 ID 생성 (취소용)
    query_id = f"{user_id}_{datetime.now().timestamp()}"
    
    # 백그라운드에서 쿼리 처리
    background_tasks.add_task(process_query_async, text, response_url, user_name, user_id, query_id, channel_id)

    # 즉시 응답 (3초 이내) - 랜덤 메시지
    start_messages = ["체크해 봐야겠군.", "움직일 시간인가."]
    return PlainTextResponse(random.choice(start_messages))

# ------------------- GPT SQL 생성 함수 -------------------
def generate_sql_with_gpt(question: str, use_full_schema: bool = False) -> str:
    # use_full_schema가 True면 구글 시트에서 전체 스키마 가져오기
    if use_full_schema:
        ctx_text = get_schema_from_google_sheets()
        if not ctx_text:
            logger.warning("⚠️ 구글 시트 실패, RAG로 폴백")
            # 구글 시트 실패 시 RAG로 폴백
            contexts = rag_retrieve(question, top_k=3)
            ctx_blocks = []
            for c in contexts:
                ctx_blocks.append(c["text"])
            ctx_text = "\n\n---\n\n".join(ctx_blocks)
    else:
        # 1) RAG로 관련 테이블 컨텍스트 가져오기
        contexts = rag_retrieve(question, top_k=3)

        ctx_blocks = []
        for c in contexts:
            # build_rag_index.py 에서 text 필드로 넣어둔 그 텍스트
            ctx_blocks.append(c["text"])
        ctx_text = "\n\n---\n\n".join(ctx_blocks)

    prompt = f"""
You are an expert MySQL assistant for the game Mafia42.

You are given context about the database schema.
Use ONLY the tables and columns that are consistent with this context.
Do NOT invent any new table or column.
Write a single valid MySQL SELECT query. No comments, no markdown, no explanation.

[CONTEXT]
{ctx_text}

[QUESTION]
{question}

[OUTPUT RULES]
- Output ONLY raw SQL.
- Do NOT wrap in ```sql ``` or any code fences.
- Use proper table and column names based on the context.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert MySQL assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    sql = response.choices[0].message.content.strip()

    # 혹시라도 말 안 듣고 ``` 붙이면 제거
    sql = sql.replace("```sql", "").replace("```", "").replace("`", "").strip()
    
    logger.info(f"✅ 생성된 SQL: {sql}")

    return sql


# ------------------- MySQL 실행 함수 -------------------
def execute_sql(sql: str):
    """SQL을 실행하고 결과를 반환. 에러 발생 시 에러 메시지 반환"""
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            cursorclass=pymysql.cursors.DictCursor
        )

        with conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()

                if not rows:
                    return "결과 없음."

                return "\n".join(str(row) for row in rows[:20])
    except pymysql.err.OperationalError as e:
        error_code, error_msg = e.args
        logger.error(f"❌ SQL 실행 오류 ({error_code}): {error_msg} | SQL: {sql}")
        return f"SQL 실행 오류 ({error_code}): {error_msg}"
    except Exception as e:
        logger.error(f"❌ 예외 발생: {str(e)} | SQL: {sql}")
        return f"오류 발생: {str(e)}"


# ------------------- Slack 메시지 전송 (피드백 버튼 포함) -------------------
def send_message_with_feedback(channel: str, text: str, question: str, sql: str, result: str):
    """피드백 버튼이 포함된 Slack 메시지 전송"""
    message_id = f"{datetime.now().timestamp()}_{hash(question + sql)}"
    
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": text
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "👍 좋아요"
                    },
                    "style": "primary",
                    "value": json.dumps({"feedback": "positive", "question": question, "sql": sql, "result": result[:500], "message_id": message_id})
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "👎 개선 필요"
                    },
                    "style": "danger",
                    "value": json.dumps({"feedback": "negative", "question": question, "sql": sql, "result": result[:500], "message_id": message_id})
                }
            ]
        }
    ]
    
    r = httpx.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        json={
            "channel": channel,
            "text": text,
            "blocks": blocks,
            "unfurl_links": False,
            "unfurl_media": False
        }
    )
    response_data = r.json()
    logger.info(f"📤 Slack 메시지 전송 완료: {response_data}")
    
    # 에러 확인
    if not response_data.get("ok"):
        logger.error(f"❌ Slack 메시지 전송 실패: {response_data.get('error', 'Unknown error')}")


def open_dm_channel(user_id: str) -> Optional[str]:
    """
    사용자와의 DM 채널 열기 (또는 기존 채널 반환)
    Slack API: https://api.slack.com/methods/conversations.open
    
    필수 권한 (Bot Token):
    - im:write (필수)
    - channels:manage (선택, 채널 관리용)
    - groups:write (선택, 그룹 DM용)
    - mpim:write (선택, 멀티 DM용)
    """
    if not user_id:
        logger.error("❌ user_id가 없어 DM 채널을 열 수 없습니다")
        return None
    
    try:
        logger.info(f"🔄 conversations.open 호출: users={user_id}")
        # users 파라미터는 단일 사용자 ID 문자열 또는 comma-separated 문자열
        # 1개만 제공하면 1:1 DM, 여러 개 제공하면 MPIM
        r = httpx.post(
            "https://slack.com/api/conversations.open",
            headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
            json={"users": user_id}  # 단일 사용자 ID 문자열
        )
        response_data = r.json()
        logger.info(f"📥 conversations.open 응답: {response_data}")
        
        if response_data.get("ok"):
            channel = response_data.get("channel", {})
            # channel이 dict인 경우 id 추출, 이미 문자열인 경우 그대로 사용
            if isinstance(channel, dict):
                channel_id = channel.get("id")
            else:
                channel_id = channel
            
            if channel_id:
                logger.info(f"✅ DM 채널 열기 성공: {channel_id}")
                return channel_id
            else:
                logger.error(f"❌ 응답에 channel.id가 없음: {response_data}")
                return None
        else:
            error = response_data.get('error', 'Unknown error')
            needed = response_data.get('needed', '')
            logger.error(f"❌ DM 채널 열기 실패: {error}")
            if needed:
                logger.error(f"   ⚠️ 필요한 권한이 없습니다: {needed}")
                logger.error(f"   Slack App의 OAuth & Permissions에서 다음 권한을 추가하세요:")
                logger.error(f"   - im:write (필수)")
                if 'channels:manage' in needed:
                    logger.error(f"   - channels:manage")
                if 'groups:write' in needed:
                    logger.error(f"   - groups:write")
                if 'mpim:write' in needed:
                    logger.error(f"   - mpim:write")
            logger.error(f"   전체 응답: {response_data}")
            return None
    except Exception as e:
        logger.error(f"❌ DM 채널 열기 예외: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def send_message(channel: str, text: str, user_id: str = None):
    """기본 Slack 메시지 전송 (피드백 버튼 없음)"""
    # user_id가 제공되고 채널이 user_id 형식이면 DM 채널 열기
    if user_id and channel.startswith("U"):
        dm_channel = open_dm_channel(user_id)
        if dm_channel:
            channel = dm_channel
    
    r = httpx.post(
        "https://slack.com/api/chat.postMessage",
        headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
        json={
            "channel": channel,
            "text": text,
            "unfurl_links": False,
            "unfurl_media": False
        }
    )
    response_data = r.json()
    logger.info(f"📤 Slack 메시지 전송: {response_data}")
    
    # 에러 확인
    if not response_data.get("ok"):
        error = response_data.get("error", "Unknown error")
        logger.error(f"❌ Slack 메시지 전송 실패: {error}")
        
        # channel_not_found 에러면 DM 채널 열기 시도
        if error == "channel_not_found" and user_id:
            logger.info(f"🔄 채널을 찾을 수 없음. DM 채널 열기 시도...")
            dm_channel = open_dm_channel(user_id)
            if dm_channel:
                # 다시 메시지 전송 시도
                r = httpx.post(
                    "https://slack.com/api/chat.postMessage",
                    headers={"Authorization": f"Bearer {SLACK_BOT_TOKEN}"},
                    json={
                        "channel": dm_channel,
                        "text": text,
                        "unfurl_links": False,
                        "unfurl_media": False
                    }
                )
                retry_data = r.json()
                logger.info(f"📤 재시도 결과: {retry_data}")
