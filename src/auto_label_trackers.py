import os
import glob
import cv2
import numpy as np

# =========================
# ====== CONFIG ==========
# =========================
FRAMES_DIR = "data/frames"                 # ROI 프레임 폴더 (jpg/png)
LABELS_DIR = "dataset/labels/train"        # YOLO 라벨 출력 폴더

TRACKER_TYPE = "csrt"   # "csrt" or "kcf" (없으면 자동 fallback)
SHOW_PREVIEW = True
EVERY_N_FRAME = 1
STOP_ON_FAIL = False

CLS_PLAYER = 0
CLS_MONSTER = 1
# =========================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def yolo_line(cls_id: int, x: float, y: float, w: float, h: float) -> str:
    return f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def clamp_box_xywh(x, y, w, h, W, H):
    x = float(max(0, min(x, W - 1)))
    y = float(max(0, min(y, H - 1)))
    w = float(max(1, min(w, W - x)))
    h = float(max(1, min(h, H - y)))
    return x, y, w, h


def xywh_to_yolo(x, y, w, h, W, H):
    xc = (x + w / 2.0) / W
    yc = (y + h / 2.0) / H
    wn = w / W
    hn = h / H
    return xc, yc, wn, hn


def _create_tracker_csrt():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    return None


def _create_tracker_kcf():
    if hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
        return cv2.legacy.TrackerKCF_create()
    return None


def make_tracker(prefer: str):
    prefer = (prefer or "").lower().strip()

    if prefer == "csrt":
        tr = _create_tracker_csrt()
        if tr is not None:
            return tr
        print("[WARN] CSRT not available. Falling back to KCF.")
        tr = _create_tracker_kcf()
        if tr is not None:
            return tr
        raise RuntimeError("Neither CSRT nor KCF trackers are available in your OpenCV build")

    if prefer == "kcf":
        tr = _create_tracker_kcf()
        if tr is not None:
            return tr
        print("[WARN] KCF not available. Falling back to CSRT.")
        tr = _create_tracker_csrt()
        if tr is not None:
            return tr
        raise RuntimeError("Neither KCF nor CSRT trackers are available in your OpenCV build")

    raise ValueError("TRACKER_TYPE must be 'csrt' or 'kcf'")


def normalize_box_xywh(box):
    """
    selectROI 반환값을 어떤 타입이든 확실히:
    - python float 4개 tuple로 변환
    """
    arr = np.asarray(box).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"ROI box must have 4 values (x,y,w,h). Got: {box}")
    arr = arr.astype(np.float32).tolist()  # 여기서 numpy 타입 제거
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def tracker_init_safe(tracker, image, box_xywh):
    """
    OpenCV 바인딩이 까다로운 환경에서 init이 박스 타입을 못 읽는 문제 회피:
    1) float tuple로 시도
    2) 실패하면 int tuple로 한 번 더 시도
    """
    b = normalize_box_xywh(box_xywh)

    try:
        ok = tracker.init(image, b)
        return ok
    except cv2.error:
        # 일부 빌드는 int를 더 잘 먹는 경우가 있음
        bi = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        ok = tracker.init(image, bi)
        return ok


def main():
    frame_paths = sorted(
        glob.glob(os.path.join(FRAMES_DIR, "*.jpg")) +
        glob.glob(os.path.join(FRAMES_DIR, "*.png"))
    )
    if not frame_paths:
        print("No frames found:", FRAMES_DIR)
        return

    ensure_dir(LABELS_DIR)

    first = cv2.imread(frame_paths[0])
    if first is None:
        print("Failed to read first frame:", frame_paths[0])
        return

    first = np.ascontiguousarray(first)

    # =========================
    # Step 1) ROI 선택
    # =========================
    print("[STEP 1] 첫 프레임에서 PLAYER 박스 선택")
    player_box = cv2.selectROI("Select PLAYER (ENTER/SPACE)", first, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if player_box is None or player_box[2] < 5 or player_box[3] < 5:
        print("Player ROI invalid or canceled. Exit.")
        return

    print("[STEP 1] MONSTER 박스들 선택 (Cancel/c/ESC 종료)")
    monster_boxes = []
    while True:
        box = cv2.selectROI("Select MONSTERs (ENTER=add, Cancel/c=finish)", first, fromCenter=False, showCrosshair=True)
        if box is None or box[2] < 5 or box[3] < 5:
            break
        monster_boxes.append(box)

    cv2.destroyAllWindows()
    print(f"Player: 1, Monsters: {len(monster_boxes)}")

    # =========================
    # Step 2) Tracker init
    # =========================
    objects = []

    tr_player = make_tracker(TRACKER_TYPE)
    ok = tracker_init_safe(tr_player, first, player_box)
    if isinstance(ok, (bool, np.bool_)) and not ok:
        raise RuntimeError("Failed to init player tracker. Try KCF or redraw ROI tighter.")
    objects.append((CLS_PLAYER, tr_player))

    for mb in monster_boxes:
        tr_m = make_tracker(TRACKER_TYPE)
        try:
            ok = tracker_init_safe(tr_m, first, mb)
        except cv2.error as e:
            print("[WARN] Monster tracker init error, skipping this box:", mb, e)
            continue
        if isinstance(ok, (bool, np.bool_)) and not ok:
            print("[WARN] Monster tracker init returned False, skipping this box:", mb)
            continue
        objects.append((CLS_MONSTER, tr_m))

    if SHOW_PREVIEW:
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)

    # =========================
    # Step 3) Track & write labels
    # =========================
    processed = 0
    failed_frames = 0

    for idx, fp in enumerate(frame_paths):
        if EVERY_N_FRAME > 1 and (idx % EVERY_N_FRAME != 0):
            continue

        frame = cv2.imread(fp)
        if frame is None:
            print("Skip unreadable:", fp)
            continue

        frame = np.ascontiguousarray(frame)
        H, W = frame.shape[:2]

        lines = []
        any_fail = False
        vis = frame.copy()

        for cls_id, trk in objects:
            ok, box = trk.update(frame)  # box: (x,y,w,h)
            if not ok:
                any_fail = True
                continue

            x, y, w, h = box
            x, y, w, h = clamp_box_xywh(x, y, w, h, W, H)

            xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, W, H)
            lines.append(yolo_line(cls_id, xc, yc, wn, hn))

            if SHOW_PREVIEW:
                color = (0, 255, 0) if cls_id == CLS_PLAYER else (0, 0, 255)
                cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

        out_txt = os.path.join(LABELS_DIR, os.path.splitext(os.path.basename(fp))[0] + ".txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        processed += 1

        if any_fail:
            failed_frames += 1
            if STOP_ON_FAIL:
                print(f"[FAIL] tracker failed at frame index={idx} file={fp} -> stopping.")
                break

        if SHOW_PREVIEW:
            cv2.putText(
                vis,
                f"{idx}/{len(frame_paths)} processed={processed} fail={any_fail} failed_total={failed_frames}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Preview", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("[STOP] user pressed q")
                break

        if processed % 200 == 0:
            print(f"processed {processed} frames... (frames_with_any_fail={failed_frames})")

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print("======================================")
    print("[DONE] YOLO label generation finished")
    print("labels_dir:", LABELS_DIR)
    print("processed:", processed)
    print("frames_with_any_tracker_fail:", failed_frames)
    print("======================================")


if __name__ == "__main__":
    main()
