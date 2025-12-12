# Osu!mania BN Helper API (Actions용, 파일 업로드 지원)

## 왜 파일 업로드를 쓰나?
GPT Actions는 request/response payload에 100,000자 제한이 있어 .osu 텍스트를 그대로 보내면 터질 수 있습니다.  
대신 Actions의 **Sending files** 기능을 쓰면, 사용자 업로드 .osu 파일이 **5분짜리 다운로드 URL**로 액션에 전달됩니다.

## 1) 로컬 실행
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export BN_HELPER_API_TOKEN="원하는토큰"  # 선택(없으면 무인증)
python app.py
```

## 2) Docker 실행(권장)
```bash
docker build -t bn-helper .
docker run -e PORT=8080 -e BN_HELPER_API_TOKEN="원하는토큰" -p 8080:8080 bn-helper
```

## 3) GPTs Builder Actions 연결
1. GPT 편집 → Configure → Actions → Add
2. `openapi.json` 붙여넣고 servers.url을 배포 URL로 교체
3. Authentication → API Key 선택 (공식 문서)
4. Key 값 = BN_HELPER_API_TOKEN 값과 동일
5. 이제 사용자에게 `.osu` 업로드를 받으면, 액션 호출 시 `openaiFileIdRefs`로 자동 전달됩니다.

## 4) 엔드포인트
- GET /health (무인증)
- POST /calc_sr (osu_text 또는 openaiFileIdRefs)
- POST /calc_pp (osu_text 또는 openaiFileIdRefs)
- POST /analyze_patterns (osu_text 또는 openaiFileIdRefs)
