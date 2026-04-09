import json
import os
import re

def clean_conversation_data():
    # Trỏ tới thư mục data của anh
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # Danh sách các file cần làm sạch
    files_to_clean = [
        "binh-thuong.jsonl", 
        "crawl-data.jsonl", 
        "tao-lao.jsonl", 
        "user_submitted_messages.jsonl"
    ]

    print("="*50)
    print("🧹 BẮT ĐẦU DỌN DẸP DỮ LIỆU...")
    print("="*50)

    for filename in files_to_clean:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue

        cleaned_entries = []
        
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    
                    # File hệ thống dùng 'full_conversation', file data lẻ dùng 'conversation'
                    text_key = "full_conversation" if "full_conversation" in item else "conversation"
                    raw_text = item.get(text_key, "")

                    # Tách đoạn chat ra thành từng dòng
                    chat_lines = raw_text.split('\n')
                    kept_lines = []
                    
                    for chat_line in chat_lines:
                        l = chat_line.strip()
                        
                        # 1. Sửa lỗi gõ nhầm "Customer: Sale: Dạ..." thành "Sale: Dạ..."
                        l = re.sub(r'^(Customer:\s*)+Sale:\s*', 'Sale: ', l)
                        
                        # 2. XÓA SẠCH CÁC DÒNG CỦA SALE
                        if l.startswith("Sale:"):
                            continue
                            
                        # 3. Dọn dẹp lỗi lặp chữ "Customer: Customer:"
                        l = re.sub(r'^(Customer:\s*)+', 'Customer: ', l)
                        
                        if l: # Nếu dòng không trống thì giữ lại
                            kept_lines.append(l)

                    # Cập nhật lại text đã dọn dẹp
                    item[text_key] = '\n'.join(kept_lines)
                    
                    # Chỉ lưu lại dòng log nếu đoạn chat vẫn còn nội dung
                    if item[text_key].strip():
                        cleaned_entries.append(item)
                        
                except Exception as e:
                    print(f"Lỗi đọc dòng trong {filename}: {e}")

        # Ghi đè lại file bằng dữ liệu siêu sạch
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in cleaned_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                
        print(f"✅ Đã dọn dẹp xong: {filename} ({len(cleaned_entries)} mẫu)")

    print("="*50)
    print("✨ DỮ LIỆU ĐÃ SẠCH! Vui lòng khởi động lại app.py để nạp Vector mới.")

if __name__ == "__main__":
    clean_conversation_data()