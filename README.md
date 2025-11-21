# TEXT_TO_SQL - Slack Bot

Slack에서 자연어 질문을 SQL 쿼리로 변환하여 데이터베이스에서 정보를 조회하는 봇입니다.

## 주요 기능

- **자연어 질문 → SQL 변환**: GPT를 사용하여 자연어를 MySQL 쿼리로 변환
- **RAG 기반 스키마 검색**: 관련 테이블 정보를 자동으로 검색
- **Google Sheets 스키마 관리**: 외부에서 스키마 정보를 관리하고 업데이트 가능
- **피드백 수집**: 사용자 피드백을 받아 쿼리 품질 개선
- **DM 및 슬래시 커맨드 지원**: DM에서 일반 메시지로 질문하거나 `/sql` 커맨드 사용
- **비동기 처리**: 긴 쿼리 실행 시 진행 상황 표시 및 취소 기능

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token

# MySQL
MYSQL_HOST=your_mysql_host
MYSQL_PORT=3306
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DB=your_database_name

# Google Sheets (선택)
GOOGLE_SHEETS_CREDENTIALS_PATH=path/to/credentials.json
GOOGLE_SHEETS_SPREADSHEET_ID=your_spreadsheet_id
GOOGLE_SHEETS_WORKSHEET_NAME=TABLE_SUMMARY
GOOGLE_SHEETS_FEEDBACK_WORKSHEET_NAME=FEEDBACK
GOOGLE_SHEETS_LOG_WORKSHEET_NAME=LOG

# RAG Index
RAG_INDEX_PATH=/path/to/rag_index.json
```

### 3. RAG Index 생성

```bash
python build_rag_index.py
```

### 4. 서버 실행

```bash
uvicorn FASTAPI:app --reload
```

## Slack App 설정

### 필수 권한 (Bot Token Scopes)

- `chat:write` - 메시지 전송
- `im:write` - DM 전송
- `im:read` - DM 읽기
- `im:history` - DM 히스토리
- `commands` - 슬래시 커맨드
- `channels:history` - 채널 메시지 읽기
- `groups:history` - 그룹 메시지 읽기

### Event Subscriptions

다음 이벤트를 구독하세요:
- `message.im` - DM 메시지
- `app_home_opened` - App Home 열림

### Slash Commands

- `/sql [질문]` - SQL 쿼리 생성 및 실행

## 사용 방법

### DM에서 사용

1. 봇과 DM 채널 열기
2. 자연어로 질문 입력 (예: "2018년 10월 29일부터 11월 30일까지 이벤트 게임 참여한 유저 카운팅")
3. 봇이 자동으로 SQL을 생성하고 실행

### 슬래시 커맨드 사용

```
/sql 유저 수 세줘
```

## 파일 구조

```
.
├── FASTAPI.py              # 메인 FastAPI 애플리케이션
├── build_rag_index.py      # RAG 인덱스 생성 스크립트
├── Data_Process.py         # 데이터 처리 유틸리티
├── resize_icon.py         # Slack 아이콘 리사이즈 유틸리티
├── requirements.txt        # Python 의존성
└── README.md              # 이 파일
```

## 주요 기능 설명

### 스키마 관리

- **RAG Index**: 로컬 JSON 파일에서 관련 테이블 정보 검색
- **Google Sheets**: 스키마 정보를 외부에서 관리 (RAG 실패 시 폴백)

### 피드백 시스템

- 사용자가 쿼리 결과에 대해 좋아요/개선 필요 피드백 제공
- 피드백은 Google Sheets에 저장되어 분석 가능

### 로그 시스템

- 모든 쿼리 실행 기록을 Google Sheets LOG 워크시트에 저장
- 피드백과 함께 쿼리 품질 추적 가능

## 라이선스

MIT

