# examples/test_ko.py

from src.ocr_main.ocr_agent import KoreanOcrAgent

# Dùng base Hàn làm fallback (nếu chưa có model fine-tune)
agent = KoreanOcrAgent(fallback_ckpt="ddobokki/ko-trocr")

print("[MODEL]", agent.used_source)

# Dự đoán 1 ảnh dòng chữ (line-level)
pred = agent.predict_image("train/images/TAF20161_00.png")
print("[PRED]", pred)

# --- (tuỳ chọn) thử nhanh recognizer PaddleOCR, nếu bạn đã cài paddleocr ---
# txt_paddle = agent.predict_image_paddle("train/images/TAF20161_00.png")
# print("[PADDLE]", txt_paddle)
