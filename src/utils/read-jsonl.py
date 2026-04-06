import json


def read_jsonl(file_path: str, deduplicate: bool = True):
    """Đọc file định dạng jsonl
    
    Args:
        file_path: đường dẫn file
        deduplicate: có loại bỏ các dòng có message trùng nhau không
    
    Returns:
        list dữ liệu đã được xử lý
    """
    data = []
    seen = set()
    duplicates_removed = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if line:
                try:
                    record = json.loads(line)
                    
                    if deduplicate and 'message' in record:
                        msg = record['message'].strip().lower()
                        if msg in seen:
                            duplicates_removed += 1
                            continue
                        seen.add(msg)
                    
                    # Chỉ giữ lại id và message
                    clean_record = {}
                    if 'id' in record:
                        clean_record['id'] = record['id']
                    if 'message' in record:
                        clean_record['message'] = record['message']
                    
                    data.append(clean_record)
                except json.JSONDecodeError:
                    continue
    
    if duplicates_removed > 0:
        print(f"ℹ️ Đã loại bỏ {duplicates_removed} dòng tin nhắn trùng lặp trong {file_path}")
    
    return data

def read_tao_lao_data(file_path: str = "data/tao-lao.jsonl"):
    """Đọc file dữ liệu tao lao định dạng jsonl (backward compatible)"""
    return read_jsonl(file_path, deduplicate=True)


def fix_ids_and_save(file_path: str = "data/tao-lao.jsonl"):
    """Sửa lại id theo thứ tự tăng dần liên tục và loại bỏ trùng lặp, ghi lại vào file"""
    data = read_jsonl(file_path, deduplicate=True)
    original_count = len(data)
    
    # Gán lại id theo thứ tự
    for index, record in enumerate(data, start=1):
        record['id'] = index
    
    # Ghi lại file
    with open(file_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record.lower(), ensure_ascii=False) + "\n")
    
    print(f"✅ Đã cập nhật {len(data)} bản ghi, id tuần tự từ 1 -> {len(data)}")


# fix_ids_and_save("data/tao-lao.jsonl")
# fix_ids_and_save("data/crawl-data.jsonl")
# fix_ids_and_save("data/binh-thuong.jsonl")
