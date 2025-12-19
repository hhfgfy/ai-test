import cv2
import numpy as np
import glob, os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_PATH = sorted(glob.glob("data/frames/*.jpg"))[0]  # 첫 프레임으로 테스트
PLAYER_TMPL = os.path.join(BASE_DIR, "..", "image", "player.png")
MONSTER_TMPL = os.path.join(BASE_DIR, "..", "image", "monster.png")

def match_one(frame_bgr, tmpl_bgr):
    # 템플릿 매칭 (정규화 상관계수)
    res = cv2.matchTemplate(frame_bgr, tmpl_bgr, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    x, y = max_loc
    h, w = tmpl_bgr.shape[:2]
    cx, cy = x + w//2, y + h//2
    return max_val, (x, y, w, h), (cx, cy)

frame = cv2.imread(FRAME_PATH)
player_t = cv2.imread(PLAYER_TMPL)
monster_t = cv2.imread(MONSTER_TMPL)

p_score, (px,py,pw,ph), (pcx,pcy) = match_one(frame, player_t)
m_score, (mx,my,mw,mh), (mcx,mcy) = match_one(frame, monster_t)

# 박스 표시
cv2.rectangle(frame, (px,py), (px+pw, py+ph), (0,255,0), 2)
cv2.putText(frame, f"PLAYER {p_score:.2f}", (px, py-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.rectangle(frame, (mx,my), (mx+mw, my+mh), (0,0,255), 2)
cv2.putText(frame, f"MONSTER {m_score:.2f}", (mx, my-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

cv2.imshow("match", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("player center:", (pcx,pcy), "score:", p_score)
print("monster center:", (mcx,mcy), "score:", m_score)
