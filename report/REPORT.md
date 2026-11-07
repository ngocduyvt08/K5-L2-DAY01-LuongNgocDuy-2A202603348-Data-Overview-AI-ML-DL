# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id=468` ; `class_name=cab` ; `rank=1` ; `score=0.510915` ; `taxonomy_name=ImageNet-1K`
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh. Vì tác vụ ở đây là `image_classification` (phân loại ảnh), dữ liệu JSON chỉ chứa kích thước chung của toàn ảnh (`image_width: 640`, `image_height: 428`) và gán nhãn dự đoán cho toàn bộ khung hình chứ khôngkhoanh vùng một vật thể cụ thể nào.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp đi kèm sẵn với taxonomy/checkpoint (cụ thể là bộ từ điển chuẩn `ImageNet-1K` gồm 1000 lớp được đóng gói trong file `yolo11n-cls.pt` khi huấn luyện). Model không thể tự nghĩ ra tên lớp mới ngoài danh sách cố định này.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Để đảm bảo tính chính xác khi xử lý tự động bằng code và đối chiếu dữ liệu:
     - `taxonomy_name` cho biết hệ quy chiếu gốc 
     - `class_id` là số nguyên cố định mà máy tính và model thực sự xuất ra 
     - `class_name` giúp con người đọc hiểu trực quan mà không cần tra bảng mã.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline cần quy định rõ quy tắc ưu tiên chọn đối tượng chính
6. Vì sao không được lấy `score` của model làm ground truth?
Vì `score` chỉ là mức độ "tự tin" xác suất toán học của riêng model dựa trên những gì nó đã học, không phải sự thật khách quan. Model hoàn toàn có thể tự tin cao vào một dự đoán sai hoặc bị bối rối khi ảnh phức tạp. Ground truth bắt buộc phải do con người kiểm chứng và gán theo quy tắc trong guideline.
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn một record.
Số lượng prediction theo ngưỡng: xem output ô so sánh 0.20 / 0.35 / 0.60 trong notebook.

1. **Chép một record:** `class_name=person` ; `score=0.912625` ; `bbox_xyxy=[385.33, 69.24, 498.92, 348.92]` ; `bbox_width=113.58` ; `bbox_height=279.68`
2. **Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**
   - Hộp giới hạn (bounding box) có điểm **góc trên-trái** nằm tại tọa độ pixel $(x_1 = 385.33, y_1 = 69.24)$ và điểm **góc dưới-phải** nằm tại tọa độ pixel $(x_2 = 498.92, y_2 = 348.92)$ (tính từ gốc $(0,0)$ ở góc trên cùng bên trái của bức ảnh kích thước $640 \times 427$).
   - Chiều rộng của hộp là $113.58\text{ pixel}$ ($498.92 - 385.33$) và chiều cao của hộp là $279.68\text{ pixel}$ ($348.92 - 69.24$).
3. **Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**
Tại ngưỡng 0.60 có 6 prediction. Tại ngưỡng 0.20 có 17 prediction.
4. **Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**
   - Khi hạ ngưỡng, model sẽ giữ lại nhiều dự đoán có độ tự tin thấp, dẫn đến tăng số lượng dương tính giả (False Positives - nhận diện nhầm cảnh nền hoặc vật thể khác) và hộp trùng lặp/dư thừa. Người Reviewer sẽ phải tốn thêm thời gian để rà soát, xóa bỏ các box rác/nhận diện sai và chỉnh sửa lại các box bị lệch thay vì chỉ kiểm tra các box chất lượng cao.
5. **Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).**
   - Quy tắc vẽ Bounding Box: Hộp giới hạn phải bao phủ át khít (tight-fitting) toàn bộ phần nhìn thấy được của vật thể (kể cả phần rìa cực đại như tay cầm, quai cốc); các cạnh của box không được cắt lẹm vào vật thể và khoảng trống thừa từ biên vật thể đến cạnh box không được vượt quá $2\text{ pixel}$.
6. **Trên ảnh kitchen, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**
   - **Vật bị cắt mép / che khuất:** Lò nướng (`oven`, tọa độ `bbox_xyxy=[0.11, 187.54, 195.96, 292.81]`) và cái bát (`bowl`, tọa độ `bbox_xyxy=[0.32, 305.02, 87.69, 371.32]`) nằm sát rìa trái bức ảnh ($x_1 \approx 0$) nên bị cắt mất một phần bởi mép ảnh và bị các đồ vật trên bàn che khuất; ngoài ra còn có một prediction `person` ở sát mép trái (`bbox_xyxy=[0.12, 263.18, 61.55, 310.78]`, `score=0.610931`) bị cắt chỉ lộ một phần nhỏ rất khó nhận diện.
   - Cách xử lý:
     - Với việc vật thể rõ ràng nhưng bị cắt mép hoặc che khuất một phần: Đưa vào Guideline thành quy tắc chung (ví dụ: *"Chỉ vẽ box bao quanh phần diện tích thực tế nhìn thấy được trong khung hình, không tự suy đoán vẽ tràn ra ngoài; bỏ qua nếu vật bị che khuất trên 80%"*) 

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn một record.

1. **Chép một record:** `instance_id=kitchen-001` ; `class_name=person` ; `score=0.899318` ; số điểm polygon=`polygon_point_count=348` ; 3–5 điểm đầu của `polygon_xy=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`
2. **So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**
   - Trong khi bounding box chỉ là một hình chữ nhật thô bao lấy 4 cực của vật thể (chứa rất nhiều pixel nền thừa xung quanh), polygon (đa giác) cung cấp đường biên chi tiết bám sát theo đúng hình dáng, đường cong và diện tích thực tế** của vật thể tới từng pixel, giúp tách biệt hoàn toàn vật thể khỏi phông nền và các vật thể lân cận.
3. **instance_id dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**
   - `instance_id` dùng để định danh duy nhất từng cá thể (đối tượng) riêng biệt trong phạm vi một bức ảnh 
   - Không phải `class_id`: Vì `class_id` chỉ cho biết loại vật thể (mọi cái bát đều có chung `class_id = 45`), không phân biệt được bát A với bát B trong cùng một ảnh.
   - Không phải `tracking_id`:Vì `tracking_id` dùng trong bài toán chuỗi video để theo dõi cùng một đối tượng xuyên suốt qua nhiều khung hình (frames) theo thời gian, còn `instance_id` ở đây chỉ có giá trị đánh số thứ tự cục bộ trong một bức ảnh tĩnh duy nhất.
4. **Viết một quy tắc vẽ biên mask để người gán nhãn làm theo.**
   - Quy tắc vẽ biên: Đặt các điểm nút (vertices) bám sát theo rìa vật lý thực tế của vật thể (sai số không quá $1-2\text{ pixel}$), tăng mật độ điểm ở các đoạn cong/góc gập để đường biên mượt và không bị cắt góc; nếu vật thể bị vật khác đè lên che khuất, chỉ vẽ viền bao quanh phần diện tích nhìn thấy được và không để mask của hai vật thể chồng lấn lên nhau.
5. **Trên ảnh kitchen, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**
   - **Vị trí trên ảnh:** Mặt bàn ăn bị đồ vật đặt phía trên đè lên và chạm sát nhau làm đường biên của mặt bàn bị cắt xẻ.

- Sử dụng Guideline để xử lý quy tắc khoét vùng che khuất giữa mặt bàn và các bát/đĩa đặt trên bàn (quy định rõ thứ tự lớp tiền cảnh/hậu cảnh: vẽ kín các vật thể nhỏ ở trên trước, phần diện tích bàn bị che sẽ tự động được trừ đi hoặc chỉ viền phần mặt bàn lộ ra).

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn lớp (`class_id`, `class_name`) cho toàn bộ ảnh | Ảnh `kitchen` nhiều đồ vật nên model đoán sai `rank=1` là `gong` | Quan sát toàn ảnh, chọn 1 lớp đúng nhất theo thứ tự ưu tiên trong guideline | Kiểm tra nhãn đã khớp chủ đề chính của ảnh và đúng quy tắc ưu tiên chưa |
| Phát hiện vật thể | Các hộp `bbox_xyxy` `[x1, y1, x2, y2]` kèm `class_id` cho từng vật | Vật bị cắt mép/nhập nhằng | Kẻ bounding box bao sát khít từng vật thể mục tiêu và gán đúng tên lớp | Soát lỗi sót vật , thừa box sai, gán sai lớp hoặc vẽ box lỏng |
| Instance segmentation | Đa giác `polygon_xy` (hoặc mask) kèm `class_id`, `instance_id` từng vật] | Vật đè nhau làm biên cắt xẻ hoặc sót | Chấm điểm đa giác bám sát viền từng cá thể, tách rõ biên các vật chạm nhau | Kiểm tra viền mask có sát mép vật không, có bị lẹm/tràn nền hay gộp nhầm vật không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Tuyệt đối không ghi họ tên, MSSV, email, số điện thoại hoặc bất kỳ dữ liệu cá nhân (PII) nào vào nội dung file `REPORT.md` và các thư mục output; thông tin họ tên/MSSV chỉ được đặt ở tên GitHub repository
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Tôi sẽ lập tức dừng việc tự ý gán nhãn và báo cáo ngay cho Lab Coach (thực hiện escalation) để xin ý kiến xử lý chứ không tự suy đoán hay tự bịa nhãn

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
