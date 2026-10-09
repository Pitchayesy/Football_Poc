Ball Detection PoC 

จับลูกบอลในคลิปถ่ายทอดสด

ใช้

Python + OpenCV + YOLOv8n (Ultralytics). โหลดน้ำหนักอัตโนมัติ.


อ่าน mp4. ยิงโมเดลทุก 2 เฟรม. เฟรมว่างใช้กล่องเก่า. ตัดช็อตทิ้งแทร็ก. กด q ออก.

รัน
```bash
pip install -r requirements.txt
python realtime_ball_poc.py
```

คลิปวาง `dataset/1.mp4`.

## จูน

`IMGSZ` ใหญ่=แม่น ช้า. `CONF` ต่ำ=เจอเยอะ false เยอะ. `INFER_EVERY` มาก=เร็ว หลุดง่าย.

