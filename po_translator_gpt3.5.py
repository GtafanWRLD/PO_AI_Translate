import os
import openai
from tqdm import tqdm
import re
import concurrent.futures
import threading
import time
from langdetect import detect, DetectorFactory
from concurrent.futures import ThreadPoolExecutor

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Optimized Config
BATCH_SIZE = 20  # Increased for efficiency
MAX_WORKERS = 6  # Increased workers
API_DELAY = 0.1  # Reduced delay
MAX_RETRIES = 3
MIN_TEXT_LENGTH = 2
MAX_TOKENS = 3800  # Close to limit but safe

# Thread-safe progress tracking
progress_lock = threading.Lock()
api_semaphore = threading.Semaphore(MAX_WORKERS)
stats_lock = threading.Lock()

# Statistics
translation_stats = {
    'total_blocks': 0,
    'english_skipped': 0,
    'russian_translated': 0,
    'empty_skipped': 0,
    'api_calls_saved': 0
}

# Simplified coordinate mapping - only what we actually need
COORD_MAP = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E',
    'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T',
    'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'Ch', 'Ш': 'Sh',
    'Щ': 'Shch', 'Ы': 'Y', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
}

# Load API key
try:
    with open("key.txt", "r", encoding="utf-8") as f:
        api_key = f.read().strip()
except FileNotFoundError:
    print("❌ key.txt not found.")
    exit(1)

client = openai.OpenAI(api_key=api_key)

def is_coordinate_text(text):
    """Check if text is ONLY a coordinate pattern like А-3, Б-5, etc."""
    if not text or not text.strip():
        return False
    
    clean_text = text.strip()
    # Must be exactly pattern: Cyrillic letter + dash + number(s)
    pattern = r'^[А-Я]-\d+$'
    return bool(re.match(pattern, clean_text))

def translate_coordinate(text):
    """Translate coordinate text directly using mapping"""
    if not is_coordinate_text(text):
        return text
    
    cyrillic_letter = text[0]
    rest = text[1:]  # "-number" part
    
    if cyrillic_letter in COORD_MAP:
        return COORD_MAP[cyrillic_letter] + rest
    
    return text  # Fallback

def contains_cyrillic(text):
    """Check if text contains Cyrillic characters"""
    if not text:
        return False
    return bool(re.search(r'[\u0400-\u04FF]', text))

def is_likely_english(text):
    """Quick check if text is already English"""
    if not text or len(text.strip()) < 3:
        return True
    
    # Remove variables and special chars for analysis
    clean = re.sub(r'%\([^)]+\)[sdifb]|%[sdifb]|\{[^}]*\}|\\[nt]', ' ', text)
    clean = re.sub(r'[^\w\s]', ' ', clean).strip().lower()
    
    if len(clean) < 3:
        return True
    
    # Common English words
    english_words = {
        'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'is', 'are',
        'battle', 'damage', 'armor', 'tank', 'destroy', 'kill', 'enemy',
        'complete', 'win', 'player', 'game', 'experience', 'points'
    }
    
    words = clean.split()
    if words:
        english_count = sum(1 for w in words if w in english_words)
        if english_count > 0 and english_count / len(words) > 0.15:
            return True
    
    # Try language detection for longer texts
    try:
        if len(clean) >= 5:
            return detect(clean) == 'en'
    except:
        pass
    
    return False

def needs_translation(text):
    """Determine if text needs translation"""
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False, 'empty'
    
    # Special handling for coordinates
    if is_coordinate_text(text):
        return True, 'coordinate'
    
    # If contains Cyrillic, needs translation
    if contains_cyrillic(text):
        return True, 'has_cyrillic'
    
    # If likely English, skip
    if is_likely_english(text):
        return False, 'english'
    
    return True, 'unknown'

def update_stats(reason, increment=1):
    """Thread-safe statistics update"""
    with stats_lock:
        if reason == 'english':
            translation_stats['english_skipped'] += increment
            translation_stats['api_calls_saved'] += increment
        elif reason in ['has_cyrillic', 'unknown', 'coordinate']:
            translation_stats['russian_translated'] += increment
        elif reason == 'empty':
            translation_stats['empty_skipped'] += increment

def read_po_file(filename):
    """Read PO file with encoding detection"""
    for encoding in ['utf-8', 'utf-8-sig', 'cp1251', 'iso-8859-1']:
        try:
            with open(filename, "r", encoding=encoding) as f:
                return f.readlines(), encoding
        except UnicodeDecodeError:
            continue
    raise Exception("Could not read file with any supported encoding")

def extract_content_from_lines(lines):
    """Extract string content from msgid/msgstr lines"""
    content = ""
    for line in lines:
        line = line.strip()
        if line.startswith('msgid ') or line.startswith('msgstr '):
            match = re.match(r'msg(?:id|str)\s+"(.*)"', line)
            if match:
                content += match.group(1)
        elif line.startswith('"') and line.endswith('"'):
            content += line[1:-1]
    
    # Unescape
    content = content.replace('\\n', '\n').replace('\\t', '\t')
    content = content.replace('\\"', '"').replace('\\\\', '\\')
    return content

def find_msgstr_blocks(lines):
    """Find msgstr blocks that need translation"""
    blocks = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('msgstr'):
            start_idx = i
            block_lines = []
            
            # Check if this is a header block (msgid "")
            is_header = False
            for j in range(i - 1, max(0, i - 10), -1):
                prev_line = lines[j].strip()
                if prev_line.startswith('msgid'):
                    if prev_line == 'msgid ""':
                        is_header = True
                    break
            
            if is_header:
                i += 1
                continue
            
            # Collect msgstr block lines
            if line == 'msgstr ""':
                block_lines.append(lines[i])
                i += 1
                while i < len(lines) and lines[i].strip().startswith('"'):
                    block_lines.append(lines[i])
                    i += 1
                end_idx = i - 1
            else:
                block_lines.append(lines[i])
                end_idx = i
                i += 1
            
            content = extract_content_from_lines(block_lines)
            
            with stats_lock:
                translation_stats['total_blocks'] += 1
            
            needs_trans, reason = needs_translation(content)
            update_stats(reason)
            
            if needs_trans:
                blocks.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'content': content,
                    'original_lines': [line.rstrip('\n\r') for line in block_lines],
                    'is_multiline': line == 'msgstr ""',
                    'reason': reason
                })
            
            continue
        
        i += 1
    
    return blocks

def translate_batch_optimized(batch_data, batch_id, progress_bar, retry_count=0):
    """Optimized batch translation with coordinate handling"""
    if not batch_data:
        return {}
    
    with api_semaphore:
        time.sleep(API_DELAY)
        
        # Separate coordinates from regular text
        coordinate_items = []
        text_items = []
        
        for item in batch_data:
            if item['reason'] == 'coordinate':
                coordinate_items.append(item)
            else:
                text_items.append(item)
        
        results = {}
        
        # Handle coordinates directly (no API call needed)
        for item in coordinate_items:
            results[item['block_idx']] = translate_coordinate(item['content'])
        
        # Handle text items with API
        if text_items:
            # Build batch text more efficiently
            batch_parts = []
            for item in text_items:
                batch_parts.append(f"ID_{item['block_idx']}: {item['content']}")
            
            batch_text = "\n---\n".join(batch_parts)
            
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a Russian-to-English translator for gaming content. "
                                "RULES:\n"
                                "1. Keep variables like %(goal)s, %d, {0} EXACTLY as-is\n"
                                "2. Keep \\n and \\t formatting\n"
                                "3. Translate Russian text to natural English\n"
                                "4. Use gaming terminology (armor, destroy, battle, tank, etc.)\n"
                                "5. Keep ID_X format and separate with ---\n"
                                "6. Be concise and direct\n\n"
                                "Format: ID_X: [translation]"
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Translate to English:\n\n{batch_text}"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=MAX_TOKENS
                )
                
                result = resp.choices[0].message.content.strip()
                
                # Parse results
                parts = result.split("---") if "---" in result else result.split("\n")
                
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    match = re.match(r'ID_(\d+):\s*(.*)', part)
                    if match:
                        block_idx = int(match.group(1))
                        translation = match.group(2).strip()
                        results[block_idx] = translation
                
                # Fallback for missing translations
                for item in text_items:
                    if item['block_idx'] not in results:
                        results[item['block_idx']] = item['content']
                
            except Exception as e:
                if retry_count < MAX_RETRIES:
                    time.sleep(1 * (retry_count + 1))
                    return translate_batch_optimized(batch_data, batch_id, progress_bar, retry_count + 1)
                else:
                    print(f"❌ Batch {batch_id} failed: {e}")
                    # Fallback to originals
                    for item in text_items:
                        results[item['block_idx']] = item['content']
        
        with progress_lock:
            progress_bar.update(len(batch_data))
        
        return results

def create_new_msgstr_lines(original_block, new_content):
    """Create new msgstr lines preserving formatting"""
    original_lines = original_block['original_lines']
    first_line = original_lines[0]
    
    # Get indentation
    indent_match = re.match(r'^(\s*)', first_line)
    indent = indent_match.group(1) if indent_match else ""
    
    new_lines = []
    
    if original_block['is_multiline'] or '\n' in new_content:
        new_lines.append(f'{indent}msgstr ""')
        
        content_lines = new_content.split('\n')
        for i, line in enumerate(content_lines):
            escaped = line.replace('\\', '\\\\').replace('"', '\\"')
            
            if i == len(content_lines) - 1:
                new_lines.append(f'{indent}"{escaped}"')
            else:
                new_lines.append(f'{indent}"{escaped}\\n"')
    else:
        escaped = new_content.replace('\\', '\\\\').replace('"', '\\"')
        new_lines.append(f'{indent}msgstr "{escaped}"')
    
    return new_lines

def process_single_file(input_path, output_path):
    """Process a single PO file"""
    print(f"📖 Processing {os.path.basename(input_path)}...")
    
    # Reset stats
    with stats_lock:
        translation_stats.update({
            'total_blocks': 0, 'english_skipped': 0, 'russian_translated': 0,
            'empty_skipped': 0, 'api_calls_saved': 0
        })
    
    lines, encoding = read_po_file(input_path)
    blocks_to_translate = find_msgstr_blocks(lines)
    
    with stats_lock:
        stats = translation_stats.copy()
    
    print(f"📊 Found {stats['total_blocks']} blocks -> {len(blocks_to_translate)} need translation")
    print(f"   🇬🇧 English (skip): {stats['english_skipped']}")
    print(f"   ⚪ Empty (skip): {stats['empty_skipped']}")
    
    if not blocks_to_translate:
        print("✅ No translation needed")
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        return 0
    
    # Add block indices
    for i, block in enumerate(blocks_to_translate):
        block['block_idx'] = i
    
    # Create batches
    batches = []
    for i in range(0, len(blocks_to_translate), BATCH_SIZE):
        batch_blocks = blocks_to_translate[i:i+BATCH_SIZE]
        batch_data = []
        for block in batch_blocks:
            batch_data.append({
                'block_idx': block['block_idx'],
                'content': block['content'],
                'reason': block['reason']
            })
        batches.append((i // BATCH_SIZE, batch_data))
    
    print(f"🚀 Processing {len(batches)} batches with {MAX_WORKERS} workers...")
    
    progress_bar = tqdm(total=len(blocks_to_translate), 
                       desc=f"Translating {os.path.basename(input_path)}", 
                       leave=False)
    
    translations = {}
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(translate_batch_optimized, batch_data, batch_id, progress_bar): 
            (batch_id, batch_data) for batch_id, batch_data in batches
        }
        
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_id, batch_data = future_to_batch[future]
            try:
                batch_results = future.result()
                translations.update(batch_results)
            except Exception as e:
                print(f"❌ Batch {batch_id} error: {e}")
                for item in batch_data:
                    translations[item['block_idx']] = item['content']
    
    progress_bar.close()
    
    # Apply translations
    new_lines = []
    current_pos = 0
    
    sorted_blocks = sorted(blocks_to_translate, key=lambda x: x['start_idx'])
    
    for block in sorted_blocks:
        start_idx = block['start_idx']
        end_idx = block['end_idx']
        block_idx = block['block_idx']
        
        # Add lines before this block
        new_lines.extend(lines[current_pos:start_idx])
        
        # Get translation
        translation = translations.get(block_idx, block['content'])
        
        # Create new msgstr lines
        new_msgstr_lines = create_new_msgstr_lines(block, translation)
        new_lines.extend([line + '\n' for line in new_msgstr_lines])
        
        current_pos = end_idx + 1
    
    # Add remaining lines
    new_lines.extend(lines[current_pos:])
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    
    return len(blocks_to_translate)

# Main execution
print("🚀 PO Translator v10.0 - Optimized & Fixed")
print(f"⚡ Config: {MAX_WORKERS} workers, batch size {BATCH_SIZE}")
print("🎯 Features: Proper coordinate detection + fast processing")
print("=" * 60)

input_dir = "input"
output_dir = "output"

if not os.path.exists(input_dir):
    os.makedirs(input_dir)
    print(f"📁 Created '{input_dir}' folder - place your .po files there")
    exit(0)

os.makedirs(output_dir, exist_ok=True)

po_files = [f for f in os.listdir(input_dir) if f.endswith(".po")]

if not po_files:
    print(f"❌ No .po files found in '{input_dir}' folder.")
    exit(1)

print(f"Found {len(po_files)} .po files:")
for fname in po_files:
    file_size = os.path.getsize(os.path.join(input_dir, fname)) / (1024*1024)
    print(f"  - {fname} ({file_size:.1f} MB)")

total_translated = 0
start_time = time.time()

for po_file in po_files:
    input_path = os.path.join(input_dir, po_file)
    output_path = os.path.join(output_dir, po_file)
    
    try:
        file_start = time.time()
        translated_count = process_single_file(input_path, output_path)
        file_time = time.time() - file_start
        
        total_translated += translated_count
        print(f"✅ {po_file} -> {translated_count} strings in {file_time:.1f}s")
    except Exception as e:
        print(f"❌ Error processing {po_file}: {e}")

total_time = time.time() - start_time

print(f"\n🎯 Translation complete!")
print("=" * 40)
print(f"📊 Files: {len(po_files)}")
print(f"📊 Strings translated: {total_translated}")
print(f"⏱️ Total time: {total_time:.1f}s")
print(f"📁 Output: '{output_dir}' folder")
