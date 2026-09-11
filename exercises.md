# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi tăng dần temperature thì câu trả lời trở nên đa dạng hơn nhưng nội dung cũng trở nên kém chính xác hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Để hỗ trợ khách hàng thì cần thông tin chính xác và ngắn gọn nên em sẽ đặt temperature thấp nhất là 0.0.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Để tính chi phí cần lấy 10000 x 3 x 350 = 10.5 triệu tokens nhân với giá của cả 2 phiên bản, mà giá của GPT-4o cao hơn khá nhiều so với GPT-4o-mini, khiến chi phí khi sử dụng GPT-4o có thể cao hơn đến hàng chục lần thậm chí cao hơn.Trường hợp dùng GPT-4o: Đối với những tác vụ cần độ chính xác cao, cũng như khả năng suy luận logic, lập trình nâng cao,... trong trường hợp mà chất lượng là ưu tiên hàng đầu.Trường hợp dùng GPT-4o-mini: Đối với những tác vụ đơn giản hơn mang tính lặp lại, cần phản hồi nhanh với chi phí thấp như phân loại thư, đọc tài liệu, tóm tắt,....

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Khi đóng vai trò giaso viên tiểu học, model sẽ trả lời ngắn hơn, dùng những từ ngữ đơn giản dễ hiểu vì đối tượng hướng đến là cho trẻ 8 tuổi. Khi đóng vai trò là một chuyên gia tài chính, câu trả lời sẽ dài hơn đáng kể, từ ngữ sử dụng sẽ là những thuật ngữ chuyên sâu cũng như là kiến thức chính xác tập trung vào khía cạnh chuyên môn nhiều hơn. Qua ví dụ này cũng thấy được là "System promt" đóng vai trò định hướng và ảnh hưởng rất nhiều đến hành vi và cách mà model sẽ giao tiếp với người dùng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Hai con số chênh nhau khoảng 90 đến 110% ( gấp khoảng 2 lần). Tiếng Việt tốn token hơn vì Tokenizer huấn luyện chủ yếu trên tiếng Anh → từ tiếng Anh thường gọn trong 1 token, còn tiếng Việt thì không có sẵn trong từ vựng đó.Chữ có dấu (à, ê, ơ, đ...) là ký tự Unicode 2 byte → BPE tách mỗi âm tiết có dấu thành 2–4 token nhỏ thay vì gộp thành 1 token. Công thức từ/0.75 chỉ đúng với tiếng Anh → áp cho tiếng Việt sẽ ước lượng thấp hơn thực tế, nên chênh lệch lớn (~2 lần).
---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong trường hợp đối thoại trực tiếp với người dùng vì nó giúp hiển thị câu trả lời trực tiếp thay vì đợi hoàn thành toàn bộ. Non-streaming phù hợp hơn trong các trường hợp xử lý dữ liệu hàng loạt hoặc tính toán chi phí token vì nó giúp đơn giản hóa việc quản lí dữ liệu.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Delay cố định khiến hàng nghìn client đồng loạt retry cùng lúc gây "thundering herd" làm server quá tải nặng hơn, trong khi exponential backoff (kèm jitter) giãn dần và rải đều thời gian retry, giảm áp lực và cho server thời gian hồi phục.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: Trợ lý hỗ trợ khách hàng cho một ứng dụng đặt lịch khám bệnh (booking app). SYSTEM PROMPT: "Bạn là trợ lý hỗ trợ khách hàng của ứng dụng đặt lịch khám bệnh MedBook. Luôn trả lời bằng tiếng Việt, giọng điệu thân thiện, lịch sự, giống một nhân viên tổng đài tận tâm. Trả lời ngắn gọn, tối đa 3 câu mỗi lượt, trừ khi người dùng yêu cầu giải thích chi tiết hơn. Không tự đưa ra chẩn đoán y khoa hay lời khuyên điều trị — nếu người dùng hỏi về triệu chứng bệnh, hãy khuyên họ đặt lịch khám với bác sĩ chuyên khoa  phù hợp. Nếu không chắc chắn về thông tin (ví dụ giá dịch vụ, lịch bác sĩ), hãy nói rõ là cần kiểm tra lại thay vì đoán bừa". Từ ngữ quan trọng trong prompt là :"Trả lời ngắn gọn, tối đa 3 câu" - Người dùng ứng dụng đặt lịch thường thao tác nhanh trên điện thoại, không muốn đọc đoạn văn dài. Giới hạn số câu buộc mô hình ưu tiên thông tin cốt lõi (ví dụ: "Bạn có thể đặt lịch vào thứ Ba lúc 9h") thay vì lan man giải thích quy trình không cần thiết, giúp trải nghiệm hội thoại giống tin nhắn thật hơn là một bài văn.
>


### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất:Không có bộ nhớ dài hạn — mỗi phiên chat độc lập, trợ lý quên hết thông tin khách hàng sau khi đóng session, khiến họ phải lặp lại thông tin mỗi lần quay lại. Cải thiện đề xuất: Xây dựng user memory store riêng (lưu trong DB gắn với user_id). Sau mỗi phiên, dùng LLM tóm tắt thông tin quan trọng thành vài dòng ngắn, lưu lại; đầu phiên mới thì truy vấn và chèn summary đó vào system prompt làm context. Định kỳ nén lại để tránh phình dung lượng, và cho phép người dùng xem/xóa dữ liệu của mình.


---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
