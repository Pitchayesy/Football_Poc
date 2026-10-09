# Ball Detection PoC v2

จับลูกบอลในคลิปถ่ายทอดสด. กล้องขยับ ซูม เบลอ ตัดช็อต ได้หมด.

## ใช้

Python + OpenCV + YOLOv8n (Ultralytics). ไม่มี train เอง. โหลดน้ำหนักอัตโนมัติ.

## ทำ

อ่าน mp4 เต็มเฟรม. ยิงโมเดลทุก 2 เฟรม. เฟรมว่างใช้กล่องเก่า. ตัดช็อตทิ้งแทร็ก. โชว์กรอบ + FPS. กด q ออก.

## รัน

```bash
pip install -r requirements.txt
python realtime_ball_poc.py
```

คลิปวาง `dataset/1.mp4`. คลิปไม่อยู่ใน git หาเอง.

## จูน

`IMGSZ` ใหญ่=แม่น ช้า. `CONF` ต่ำ=เจอเยอะ false เยอะ. `INFER_EVERY` มาก=เร็ว หลุดง่าย.

วัดคลิปไฮไลต์ 720p: เจอ 137/150 เฟรม (91%).
