import json


def read_jsonl(file_path: str, deduplicate: bool = True):
    """Đọc file jsonl có cấu trúc conversation (dùng cho cả 3 tập: tao-lao, binh-thuong, crawl-data)
    
    Args:
        file_path: đường dẫn file jsonl
        deduplicate: có loại bỏ các hội thoại trùng lặp không (dựa trên nội dung conversation)
    
    Returns:
        list of dict: mỗi dict chứa id, predicted_label, conversation
    """
    data = []
    seen = set()
    duplicates_removed = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
                
                # Lấy nội dung conversation để kiểm tra trùng lặp
                conv_text = record.get('conversation', '').strip()
                
                if deduplicate and conv_text:
                    if conv_text in seen:
                        duplicates_removed += 1
                        continue
                    seen.add(conv_text)
                
                # Chỉ giữ lại các trường cần thiết
                clean_record = {
                    'id': record.get('id'),
                    'predicted_label': record.get('predicted_label'),
                    'conversation': conv_text
                }
                
                # Loại bỏ nếu không có conversation
                if clean_record['conversation']:
                    data.append(clean_record)
                    
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc dòng: {e}")
                continue
    
    if duplicates_removed > 0:
        print(f"ℹ️ Đã loại bỏ {duplicates_removed} hội thoại trùng lặp trong {file_path}")
    
    print(f"✅ Đã đọc {len(data)} bản ghi từ {file_path}")
    return data


def read_tao_lao_data(file_path: str = "data/tao-lao.jsonl"):
    """Backward compatible - Đọc file tao-lao.jsonl"""
    return read_jsonl(file_path, deduplicate=True)


def read_binh_thuong_data(file_path: str = "data/binh-thuong.jsonl"):
    """Đọc file binh-thuong.jsonl"""
    return read_jsonl(file_path, deduplicate=True)


def read_crawl_data(file_path: str = "data/crawl-data.jsonl"):
    """Đọc file crawl-data.jsonl"""
    return read_jsonl(file_path, deduplicate=True)


def fix_ids_and_save(file_path: str):
    """Sửa lại id theo thứ tự tăng dần và ghi đè file"""
    data = read_jsonl(file_path, deduplicate=True)
    original_count = len(data)
    
    # Gán lại id theo thứ tự
    for index, record in enumerate(data, start=1):
        record['id'] = index
    
    # Ghi lại file
    with open(file_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"✅ Đã cập nhật {len(data)} bản ghi (id từ 1 → {len(data)}) cho file: {file_path}")
    return len(data)


# ====================== CHẠY THỦ CÔNG NẾU CẦN ======================
if __name__ == "__main__":
    print("🔧 Công cụ sửa id và đọc file jsonl\n")
    
    # Uncomment dòng nào bạn muốn chạy
    # fix_ids_and_save("data/tao-lao.jsonl")
    # fix_ids_and_save("data/binh-thuong.jsonl")
    # fix_ids_and_save("data/crawl-data.jsonl")
    
    # Test đọc dữ liệu
    # data = read_jsonl("data/tao-lao.jsonl")
    # print(f"Đọc được {len(data)} hội thoại từ tao-lao.jsonl")