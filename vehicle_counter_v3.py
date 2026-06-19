import cv2
import sqlite3
import yaml
from datetime import date, datetime
from ultralytics import YOLO


VIDEO_SOURCE   = 0
LINE_POSITION  = 0.5      
ZONE_MARGIN    = 35       
DB_FILE        = "vehicle_counts.db"
MODEL_FILE     = "yolov8n.pt"
CONF_THRESHOLD = 0.25      
IMG_SIZE       = 640       

CAR_ID        = 2
MOTORCYCLE_ID = 3


TRACKER_CONFIG_FILE = "custom_tracker.yaml"

def write_custom_tracker_config():
    config = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.5,
        "track_buffer": 60,       
        "match_thresh": 0.75,     
        "fuse_score": True,
    }
    with open(TRACKER_CONFIG_FILE, "w") as f:
        yaml.dump(config, f)


def init_db():
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS counts (
            date        TEXT PRIMARY KEY,
            cars        INTEGER DEFAULT 0,
            motorcycles INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crossings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            date        TEXT,
            time        TEXT,
            track_id    INTEGER,
            class       TEXT,
            custom_name TEXT DEFAULT NULL
        )
    """)
    conn.commit()
    return conn

def save_daily_counts(conn, cars, motorcycles):
    today = str(date.today())
    conn.execute("""
        INSERT INTO counts (date, cars, motorcycles) VALUES (?, ?, ?)
        ON CONFLICT(date) DO UPDATE
        SET cars=excluded.cars, motorcycles=excluded.motorcycles
    """, (today, cars, motorcycles))
    conn.commit()

def save_crossing(conn, track_id, class_name):
    now = datetime.now()
    conn.execute("""
        INSERT INTO crossings (date, time, track_id, class, custom_name)
        VALUES (?, ?, ?, ?, NULL)
    """, (str(now.date()), now.strftime("%H:%M:%S"), track_id, class_name))
    conn.commit()

def load_today_counts(conn):
    today = str(date.today())
    row = conn.execute(
        "SELECT cars, motorcycles FROM counts WHERE date=?", (today,)
    ).fetchone()
    return (row[0], row[1]) if row else (0, 0)

def get_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def main():
    write_custom_tracker_config()  

    model = YOLO(MODEL_FILE)
    cap   = cv2.VideoCapture(VIDEO_SOURCE)
    conn  = init_db()

    car_count, moto_count = load_today_counts(conn)
    crossed_ids  = set()   
    track_sides  = {}      

    print(" Zone-based counting + tuned tracker active.")
    print("Starting vehicle counter... Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w     = frame.shape[:2]
        line_y   = int(h * LINE_POSITION)
        zone_top = line_y - ZONE_MARGIN
        zone_bot = line_y + ZONE_MARGIN

        
        results = model.track(
            frame, persist=True, verbose=False,
            classes=[CAR_ID, MOTORCYCLE_ID],
            tracker=TRACKER_CONFIG_FILE,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE
        )

        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            trk_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, cls_id, trk_id in zip(boxes, cls_ids, trk_ids):
                cx, cy = get_center(box)

                class_name = "Car" if cls_id == CAR_ID else "Motorcycle"
                label      = f"{class_name} #{trk_id}"
                color      = (0, 200, 255) if cls_id == CAR_ID else (255, 100, 0)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)

               
                if cy < zone_top:
                    side = "above"
                elif cy > zone_bot:
                    side = "below"
                else:
                    side = None  

                last_side = track_sides.get(trk_id)

                
                if (side == "below" and last_side == "above"
                        and trk_id not in crossed_ids):
                    crossed_ids.add(trk_id)

                    if cls_id == CAR_ID:
                        car_count += 1
                    else:
                        moto_count += 1

                    save_daily_counts(conn, car_count, moto_count)
                    save_crossing(conn, int(trk_id), class_name)

                    print(f"  [{label}] crossed! "
                          f"Cars: {car_count} | Motos: {moto_count}")

                
                if side is not None:
                    track_sides[trk_id] = side

        
        cv2.rectangle(frame, (0, zone_top), (w, zone_bot), (0, 255, 0), 1)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 0), 2)
        cv2.putText(frame, "COUNTING ZONE", (10, zone_top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.rectangle(frame, (0, 0), (270, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Cars       : {car_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Motorcycles: {moto_count}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        cv2.imshow("University Vehicle Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    save_daily_counts(conn, car_count, moto_count)
    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    print(f"\nDone — Cars: {car_count} | Motorcycles: {moto_count}")

if __name__ == "__main__":
    main()
