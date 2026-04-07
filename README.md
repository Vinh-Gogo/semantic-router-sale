# SALE ANALYTICS - Semantic Router

## Instructions install

> - *Docker enpoints url*.
> - *Data source*.
> - *GPU or CPU*.
> - *Environment `uv`*

## Quick Test

```bash
# At root directory
uv run uvicorn visualization.app:app --reload
or
python -m uvicorn visualization.app:app --host 127.0.0.1 --port 8000 --reload
```

---

✅ **Đã hiểu rõ vai trò của anh:**  
Anh là **nhân viên sale cho thuê căn hộ dịch vụ** (serviced apartment), và anh muốn nắm rõ **chủ đề chính** của từng loại file JSONL để sau này dùng cho training AI, filter lead, hoặc xây chatbot xử lý khách hàng.

Dưới đây là phân tích chi tiết **chủ đề liên quan** của từng file, được thiết kế phù hợp với ngữ cảnh **bán căn hộ dịch vụ** (ngắn hạn/dài hạn, khách người Việt & nước ngoài, khu vực HCM):

### 1. **binh-thuong.jsonl**  
**Tên đầy đủ:** Bình thường / Lịch sự / Khách hàng chuẩn  
**Đặc điểm:** Tin nhắn chuyên nghiệp, rõ ràng, mang tính hỏi thông tin thực tế. Đây là 70-80% lead thật mà sale thường gặp.

**Chủ đề chính:**
- Hỏi thông tin căn hộ (diện tích, số phòng ngủ, nội thất, view)
- Giá thuê, chi phí phát sinh (điện nước, internet, phí quản lý, gửi xe)
- Vị trí & tiện ích xung quanh (gần công ty, bệnh viện, siêu thị, metro, sân bay)
- Thời gian thuê (ngắn hạn 1-3 tháng hay dài hạn)
- Thủ tục thuê (cọc, hợp đồng, giấy tờ, check-in/check-out)
- Tiện ích căn hộ dịch vụ (máy giặt, bếp, máy lạnh, wifi tốc độ cao, hồ bơi, gym, an ninh 24/7)
- So sánh giá với các dự án khác
- Hỏi lịch xem nhà / hẹn lịch
- Yêu cầu ảnh/video thực tế, bản đồ, hợp đồng mẫu
- Khách hỏi về quy định chung cư (không hút thuốc, không thú cưng, giờ giấc…)

### 2. **crawl-data.jsonl**  
**Tên đầy đủ:** Dữ liệu crawl từ web/Facebook/group  
**Đặc điểm:** Dữ liệu thô thu thập từ các nguồn công khai (group “Cho thuê căn hộ HCM”, “Căn hộ dịch vụ Quận 1,7,2”, Fanpage, website batdongsan, chotot, etc.). Có cả tin đăng cho thuê lẫn comment của khách thật.

**Chủ đề chính:**
- Bài đăng cho thuê thực tế (mô tả căn hộ, giá, liên hệ)
- Comment hỏi giá, hỏi còn phòng không, hỏi giảm giá
- Review của khách cũ (tốt/xấu về vệ sinh, tiếng ồn, chủ nhà, quản lý)
- So sánh căn hộ dịch vụ với chung cư thông thường
- Hỏi về “căn studio”, “căn duplex”, “căn full nội thất”
- Thảo luận về khu vực hot (Thảo Điền, Quận 7, Phú Mỹ Hưng, Bến Thành, Quận 1)
- Khách chia sẻ kinh nghiệm thuê căn hộ dịch vụ (lần đầu thuê, thuê cho sếp nước ngoài…)
- Tin đăng kèm ảnh thật + video quay phòng
- Câu hỏi về phí ẩn, hợp đồng điện tử, đặt cọc online

### 3. **tao-lao.jsonl**  
**Tên đầy đủ:** Tào lao / Biến thái / Khách “khó” hoặc có ý đồ  
**Đặc điểm:** Tin nhắn lộn xộn, nhảm, tục tĩu, tán tỉnh, hoặc ngầm đòi “dịch vụ kèm theo”. Đây là nhóm lead mà sale cần lọc hoặc xử lý khéo (khoảng 10-15% nhưng gây tốn thời gian nhất).

**Chủ đề chính (rất sát với 2 file anh đã tạo trước):**
- Tán tỉnh biến thái với sale (nữ): “Em xinh quá, anh thuê căn hộ để gặp em”, “Em mặc đồ sexy đi xem nhà với anh nha”
- Yêu cầu “dịch vụ đặc biệt” ngầm: gái gọi, massage, party, “căn hộ có thêm dịch vụ”, “cho phép dẫn bạn gái khác nhau”
- Hỏi về việc dùng căn hộ để “nhậu nhẹt, đánh bạc, hút chích”
- Tin nhắn tào lao kiểu: “Căn hộ có cho ở chung với 5-6 người không?”, “Có cho quay phim không?”, “Có camera trong phòng không?”
- Chửi bới, đòi giảm giá kiểu “anh là VIP, anh giới thiệu nhiều khách”
- Ngáo đá / say rượu nhắn linh tinh
- Hỏi giá kiểu “thuê 1 đêm được không”, “thuê giờ được không”
- Yêu cầu “giữ bí mật”, “không ghi tên thật”, “không check giấy tờ”
- Kết hợp tệ nạn: “Căn hộ yên tĩnh không, anh hay tổ chức tiệc”, “Có cho hút thuốc lá trong phòng không?”