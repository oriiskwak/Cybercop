"""
사기 증거 어휘 — v2 ALLOWED_OBJECTS 사기 특화 항목만 추출.
DINO는 영어 학습 기반이므로 {한국어 레이블: 영어 DINO 쿼리} 형태로 관리.
탐지 결과는 영어 → 한국어로 역매핑해서 반환.
"""

SCAM_EVIDENCE_VOCAB: dict[str, str] = {
    # ── 화면/UI ──────────────────────────────────────────────────────────────
    "주식 차트":           "stock chart",
    "캔들 차트":           "candlestick chart",
    "수익률 그래프":       "profit rate graph",
    "암호화폐 차트":       "cryptocurrency chart",
    "종목 추천 화면":      "stock recommendation screen",
    "수익 인증 화면":      "profit verification screen",
    "카카오톡 화면":       "KakaoTalk chat screen",
    "텔레그램 화면":       "Telegram chat screen",
    "채팅방 화면":         "chat room screen",
    "오픈채팅방":          "open chat room",
    "문자 메시지 화면":    "text message screen",
    "계좌 화면":           "bank account screen",
    "계좌번호":            "account number",
    "입출금 내역":         "transaction history",
    "송금 화면":           "money transfer screen",
    "인터넷 뱅킹 화면":   "internet banking screen",
    "결제 화면":           "payment screen",
    "인증 화면":           "authentication screen",
    "비트코인 화면":       "bitcoin screen",
    "가상화폐 지갑 화면":  "cryptocurrency wallet screen",
    "틱톡 화면":           "TikTok screen",
    "뉴스 화면":           "news screen",
    "재택 수익 인증 화면": "remote income proof screen",
    "개인정보 입력 화면":  "personal information input screen",
    "포인트 적립 화면":    "points reward screen",
    # ── 물체/증거 ────────────────────────────────────────────────────────────
    "QR 코드":             "QR code",
    "딥페이크 얼굴":       "deepfake face",
    "신분증":              "ID card",
    "가짜 신분증":         "fake ID card",
    "검찰청 공문서":       "official prosecution document",
    "계약서":              "contract document",
    "가품 제품":           "counterfeit product",
    "명품 로고":           "luxury brand logo",
    "제품 태그":           "product tag",
    "출금 화면":           "cash withdrawal screen",
    "ATM 화면":            "ATM screen",
}

# 이모티콘·아이콘 — 2026-10 통합본. 판정에는 쓰지 않고, 사기/검토필요 판정 뒤 표시용으로만 closed-set 탐지.
EMOTICON_VOCAB: list[str] = [
    "하트", "웃는 얼굴", "별", "빨간색 이모지", "따봉", "별 모양 이모지 2개", "불꽃", "우는 얼굴 이모지", "화난 얼굴",
]

ICON_VOCAB: list[str] = [
    "프로필 사진", "닫기", "화살표", "검색", "메뉴", "설정", "댓글 아이콘", "확장 버튼", "뒤로 가기", "알림",
    "카카오톡 로고", "체크 표시", "좋아요 버튼", "VITA 로고", "전송 버튼", "최소화", "No.1 로고", "최대화 아이콘",
    "공유 버튼", "입력창 전송 버튼", "월계수 관 로고", "홈", "별 아이콘", "플러스 버튼", "체크박스", "채팅창 UI 아이콘",
    "창 크기 조절 버튼", "로고", "스피커 아이콘", "달러 기호", "종 모양", "창 아이콘", "카메라 아이콘", "마이크 아이콘",
    "스크롤 바 아이콘", "카카오톡 검색 아이콘", "카카오톡 채팅 아이콘", "카카오톡 메뉴 아이콘", "카카오톡 설정 아이콘",
    "앱 아이콘", "이전 화살표", "상승 화살표", "알림 종 아이콘", "다음 화살표", "하단 메뉴 아이콘", "파워볼 로고",
    "메시지 말풍선 아이콘", "사진 아이콘", "윈도우 창 닫기 버튼", "채팅 말풍선 아이콘",
]
