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
