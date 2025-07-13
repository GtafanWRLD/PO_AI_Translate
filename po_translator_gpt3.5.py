import os
import json
import openai
from tqdm import tqdm
import re
import concurrent.futures
import threading
import time
from langdetect import detect, DetectorFactory
from concurrent.futures import ThreadPoolExecutor
import sys

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configuration file name
CONFIG_FILE = "config.json"

# Default configuration (WITHOUT language settings)
DEFAULT_CONFIG = {
    "batch_size": 20,
    "max_workers": 6,
    "api_delay": 0.1,
    "max_retries": 3,
    "max_tokens": 3800,
    "min_text_length": 2
}

# Language codes mapping for better user experience
LANGUAGE_CODES = {
    "EN": "English",
    "ES": "Spanish", 
    "FR": "French",
    "DE": "German",
    "IT": "Italian",
    "PT": "Portuguese",
    "PL": "Polish",
    "NL": "Dutch",
    "SV": "Swedish",
    "NO": "Norwegian",
    "DA": "Danish",
    "FI": "Finnish",
    "CS": "Czech",
    "SK": "Slovak",
    "HU": "Hungarian",
    "RO": "Romanian",
    "BG": "Bulgarian",
    "HR": "Croatian",
    "SR": "Serbian",
    "SL": "Slovenian",
    "ET": "Estonian",
    "LV": "Latvian",
    "LT": "Lithuanian",
    "EL": "Greek",
    "TR": "Turkish",
    "AR": "Arabic",
    "HE": "Hebrew",
    "JA": "Japanese",
    "KO": "Korean",
    "ZH": "Chinese",
    "TH": "Thai",
    "VI": "Vietnamese",
    "HI": "Hindi",
    "UR": "Urdu"
}

# Language detection mapping (langdetect codes to our codes)
LANGDETECT_TO_CODE = {
    'en': 'EN',
    'es': 'ES', 
    'fr': 'FR',
    'de': 'DE',
    'it': 'IT',
    'pt': 'PT',
    'pl': 'PL',
    'nl': 'NL',
    'sv': 'SV',
    'no': 'NO',
    'da': 'DA',
    'fi': 'FI',
    'cs': 'CS',
    'sk': 'SK',
    'hu': 'HU',
    'ro': 'RO',
    'bg': 'BG',
    'hr': 'HR',
    'sr': 'SR',
    'sl': 'SL',
    'et': 'ET',
    'lv': 'LV',
    'lt': 'LT',
    'el': 'EL',
    'tr': 'TR',
    'ar': 'AR',
    'he': 'HE',
    'ja': 'JA',
    'ko': 'KO',
    'zh': 'ZH',
    'th': 'TH',
    'vi': 'VI',
    'hi': 'HI',
    'ur': 'UR'
}

# Thread-safe progress tracking
progress_lock = threading.Lock()
api_semaphore = None
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

def create_directories():
    """Create input and output directories if they don't exist"""
    directories = ["input", "output"]
    created = []
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            created.append(directory)
    
    return created

def load_config():
    """Load configuration from file or create default (WITHOUT language settings)"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Remove language settings if they exist in the config file
            config.pop('target_language', None)
            config.pop('target_language_name', None)
            
            print(f"✅ Loaded configuration from {CONFIG_FILE}")
            return config
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            print("Using default configuration...")
            return DEFAULT_CONFIG.copy()
    else:
        return None

def save_config(config):
    """Save configuration to file (WITHOUT language settings)"""
    try:
        # Create a copy of config without language settings
        config_to_save = config.copy()
        config_to_save.pop('target_language', None)
        config_to_save.pop('target_language_name', None)
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=4, ensure_ascii=False)
        print(f"✅ Configuration saved to {CONFIG_FILE} (language settings excluded)")
    except Exception as e:
        print(f"❌ Error saving config: {e}")

def get_user_input(prompt, default=None, input_type=str, validation_func=None):
    """Get user input with validation"""
    while True:
        if default is not None:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        try:
            # Convert to required type
            if input_type == int:
                value = int(user_input)
            elif input_type == float:
                value = float(user_input)
            else:
                value = user_input
            
            # Apply validation if provided
            if validation_func:
                if validation_func(value):
                    return value
                else:
                    print("❌ Invalid input. Please try again.")
            else:
                return value
                
        except ValueError:
            print(f"❌ Please enter a valid {input_type.__name__}")

def setup_language():
    """Setup target language with user-friendly interface"""
    print("\n🌍 LANGUAGE SELECTION")
    print("=" * 50)
    print("Popular languages:")
    popular = ["EN", "ES", "FR", "DE", "IT", "PT", "PL", "NL"]
    
    for i, code in enumerate(popular, 1):
        print(f"  {i:2}. {code} - {LANGUAGE_CODES[code]}")
    
    print(f"\n📝 You can also enter any language code manually")
    print("   Examples: JA (Japanese), KO (Korean), ZH (Chinese), etc.")
    
    while True:
        choice = input("\nEnter language code or number (1-8): ").strip().upper()
        
        # Check if it's a number choice
        try:
            num_choice = int(choice)
            if 1 <= num_choice <= len(popular):
                code = popular[num_choice - 1]
                return code, LANGUAGE_CODES[code]
        except ValueError:
            pass
        
        # Check if it's a direct language code
        if len(choice) == 2 and choice.isalpha():
            if choice in LANGUAGE_CODES:
                return choice, LANGUAGE_CODES[choice]
            else:
                # Allow custom language codes
                lang_name = input(f"Language name for '{choice}': ").strip()
                if lang_name:
                    return choice, lang_name
        
        print("❌ Invalid input. Please enter a number (1-8) or 2-letter language code.")

def initial_setup():
    """First-time setup wizard"""
    print("🚀 PO TRANSLATOR - FIRST TIME SETUP")
    print("=" * 60)
    print("Welcome! Let's configure the translator for optimal performance.")
    
    config = DEFAULT_CONFIG.copy()
    
    # Performance settings
    print("\n⚡ PERFORMANCE SETTINGS")
    print("=" * 30)
    print("These settings affect translation speed and API usage:")
    
    config["batch_size"] = get_user_input(
        "📦 Batch size (texts per API call, 10-50 recommended)",
        default=20,
        input_type=int,
        validation_func=lambda x: 5 <= x <= 100
    )
    
    config["max_workers"] = get_user_input(
        "👥 Max workers (parallel threads, 3-10 recommended)", 
        default=6,
        input_type=int,
        validation_func=lambda x: 1 <= x <= 20
    )
    
    config["api_delay"] = get_user_input(
        "⏳ API delay in seconds (0.1-1.0 recommended)",
        default=0.1,
        input_type=float,
        validation_func=lambda x: 0.05 <= x <= 5.0
    )
    
    # Advanced settings
    advanced = input("\n🔧 Configure advanced settings? (y/N): ").strip().lower()
    if advanced == 'y':
        print("\n🔧 ADVANCED SETTINGS")
        print("=" * 20)
        
        config["max_retries"] = get_user_input(
            "🔄 Max retries for failed requests",
            default=3,
            input_type=int,
            validation_func=lambda x: 1 <= x <= 10
        )
        
        config["max_tokens"] = get_user_input(
            "🎯 Max tokens per API call",
            default=3800,
            input_type=int,
            validation_func=lambda x: 1000 <= x <= 4000
        )
        
        config["min_text_length"] = get_user_input(
            "📏 Minimum text length to translate",
            default=2,
            input_type=int,
            validation_func=lambda x: 1 <= x <= 10
        )
    
    # Save configuration (without language settings)
    save_config(config)
    
    print(f"\n🎯 Setup complete! Configuration saved to {CONFIG_FILE}")
    print("💡 Note: Language selection will be prompted each time you run the program.")
    print("    Other settings can be modified by editing the config file.")
    
    return config

def check_api_key():
    """Check if API key exists and is valid"""
    api_key_file = "key.txt"
    
    if not os.path.exists(api_key_file):
        print(f"\n❌ API key file '{api_key_file}' not found!")
        print("Please create a 'key.txt' file with your OpenAI API key.")
        print("Example: echo 'sk-your-api-key-here' > key.txt")
        return None
    
    try:
        with open(api_key_file, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        
        if not api_key:
            print(f"❌ API key file '{api_key_file}' is empty!")
            return None
        
        if not api_key.startswith('sk-'):
            print("⚠️ Warning: API key doesn't look like an OpenAI key (should start with 'sk-')")
        
        return api_key
        
    except Exception as e:
        print(f"❌ Error reading API key: {e}")
        return None

def detect_language_enhanced(text):
    """
    Enhanced language detection with better accuracy
    Returns language code (EN, RU, PL, etc.) or 'unknown'
    """
    if not text or len(text.strip()) < 3:
        return 'unknown'
    
    try:
        # Clean text for detection - remove variables and formatting but keep more text
        clean_text = re.sub(r'%\([^)]+\)s|%\w+|{\w+}|\\[ntr]', ' ', text)
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)  # Remove special chars but keep letters
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) < 3:
            return 'unknown'
        
        # Use langdetect with confidence check
        detected = detect(clean_text)
        
        # Convert langdetect code to our format
        our_code = LANGDETECT_TO_CODE.get(detected, 'unknown')
        
        # Additional checks for common cases
        if our_code == 'unknown':
            # Check for Cyrillic (Russian/Ukrainian)
            if re.search(r'[\u0400-\u04FF]', text):
                # More specific detection between RU and UK
                if re.search(r'[іїєґ]', text):  # Ukrainian specific chars
                    return 'UK'
                else:
                    return 'RU'
            # Check for Polish specific characters
            elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
                return 'PL'
            # Check for basic Latin
            elif re.search(r'[a-zA-Z]', text):
                return 'EN'  # Default to English for Latin script
        
        return our_code
        
    except Exception as e:
        # Fallback to regex-based detection
        if re.search(r'[\u0400-\u04FF]', text):
            if re.search(r'[іїєґ]', text):
                return 'UK'
            else:
                return 'RU'
        elif re.search(r'[ąćęłńóśźż]', text, re.IGNORECASE):
            return 'PL'
        elif re.search(r'[a-zA-Z]', text):
            return 'EN'
        return 'unknown'

def should_translate_content(content, target_language):
    """
    Enhanced logic to determine if content should be translated
    Returns (should_translate: bool, reason: str, detected_lang: str)
    """
    if not content or not content.strip():
        return False, 'empty', 'unknown'
    
    # Check minimum length
    if len(content.strip()) < config.get('min_text_length', 2):
        return False, 'too_short', 'unknown'
    
    # Detect language
    detected_lang = detect_language_enhanced(content)
    
    # Check if it's coordinate format
    if is_coordinate_format(content):
        return False, 'coordinate', detected_lang
    
    # Check for patterns that shouldn't be translated
    skip_patterns = [
        r'^[A-Z]\d+$',  # Coordinate patterns
        r'^\d+$',       # Pure numbers
        r'^[.,:;!?]+$', # Pure punctuation
        r'^[A-Za-z0-9_]+\.(png|jpg|jpeg|gif|svg|wav|mp3|ogg)$',  # File names
        r'^#[0-9A-Fa-f]{6}$',  # Hex colors
        r'^\w+://',     # URLs
        r'^[A-Za-z0-9_]+$',  # Single words that might be identifiers
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, content.strip()):
            return False, 'skip_pattern', detected_lang
    
    # Main logic: translate if detected language is different from target
    if detected_lang == 'unknown':
        # If we can't detect, assume it needs translation (conservative approach)
        return True, 'unknown_assume_translate', detected_lang
    elif detected_lang == target_language:
        # Same language as target - don't translate
        return False, 'same_as_target', detected_lang
    else:
        # Different language - translate it
        return True, 'different_language', detected_lang

def is_coordinate_format(text):
    """Check if text is a coordinate format like A1, B2, etc."""
    if not text or len(text.strip()) > 10:
        return False
    
    text = text.strip()
    
    # Check for simple coordinate patterns
    if re.match(r'^[A-Z]\d+$', text):
        return True
    
    # Check for Cyrillic coordinates
    if re.match(r'^[А-Я]\d+$', text):
        return True
    
    # Check for coordinate text like А-3, Б-5
    if re.match(r'^[А-Я]-\d+$', text):
        return True
    
    return False

def translate_coordinate(coord_text):
    """Transliterate Cyrillic coordinates to Latin"""
    if not coord_text:
        return coord_text
    
    result = ""
    for char in coord_text:
        if char in COORD_MAP:
            result += COORD_MAP[char]
        else:
            result += char
    
    return result

def is_coordinate_text(text):
    """Check if text is ONLY a coordinate pattern like А-3, Б-5, etc."""
    if not text or not text.strip():
        return False
    
    clean_text = text.strip()
    # Must be exactly pattern: Cyrillic letter + dash + number(s)
    pattern = r'^[А-Я]-\d+$'
    return bool(re.match(pattern, clean_text))

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
    """Determine if text needs translation - LEGACY FUNCTION for compatibility"""
    should_translate, reason, detected_lang = should_translate_content(text, config.get('target_language', 'EN'))
    
    # Map new reasons to old reasons for compatibility
    if reason in ['same_as_target', 'english']:
        return False, 'english'
    elif reason == 'empty':
        return False, 'empty'
    elif reason == 'coordinate':
        return True, 'coordinate'
    elif reason in ['different_language', 'has_cyrillic', 'unknown_assume_translate']:
        return True, 'has_cyrillic'
    else:
        return should_translate, reason

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

def parse_po_string(lines):
    """Parse PO string content handling proper unescaping"""
    content = ""
    for line in lines:
        line = line.strip()
        if line.startswith('msgid ') or line.startswith('msgstr '):
            # Extract quoted content after msgid/msgstr
            match = re.match(r'msg(?:id|str)\s+"(.*)"', line)
            if match:
                content += match.group(1)
        elif line.startswith('"') and line.endswith('"'):
            # Multi-line string continuation
            content += line[1:-1]
    
    # Proper unescaping - order matters!
    content = content.replace('\\n', '\n')
    content = content.replace('\\t', '\t')
    content = content.replace('\\r', '\r')
    content = content.replace('\\"', '"')
    content = content.replace('\\\\', '\\')
    
    return content

def escape_po_string(text):
    """Properly escape text for PO format"""
    # Order matters for escaping!
    text = text.replace('\\', '\\\\')  # Must be first
    text = text.replace('"', '\\"')
    text = text.replace('\r', '\\r')
    text = text.replace('\t', '\\t')
    # Don't escape \n here - we handle it specially in create_new_msgstr_lines
    return text

def unescape_po(text):
    """Unescape PO format text back to original"""
    if not text:
        return ""
    
    # Reverse the escaping process
    unescaped = text.replace('\\"', '"')  # Unescape quotes first
    unescaped = unescaped.replace('\\\\', '\\')  # Then unescape backslashes
    
    return unescaped

def extract_content_from_lines(lines):
    """Extract string content from msgid/msgstr lines with proper parsing"""
    return parse_po_string(lines)

def find_msgstr_blocks(lines):
    """Find msgstr blocks that need translation with enhanced language detection"""
    blocks = []
    i = 0
    stats = {
        'empty': 0, 'too_short': 0, 'same_as_target': 0, 
        'coordinate': 0, 'skip_pattern': 0, 'different_language': 0,
        'unknown_assume_translate': 0
    }
    lang_stats = {}
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('msgstr'):
            start_idx = i
            block_lines = []
            
            # Check if this is a header block or comment block
            is_header = False
            is_comment_block = False
            
            # Look backwards to find the corresponding msgid
            for j in range(i - 1, max(0, i - 20), -1):
                prev_line = lines[j].strip()
                if prev_line.startswith('msgid'):
                    if prev_line == 'msgid ""':
                        is_header = True
                    break
                elif prev_line.startswith('#'):
                    # Check if this is a special comment that shouldn't be translated
                    if 'PluralForms' in prev_line or 'plural=' in prev_line:
                        is_comment_block = True
            
            if is_header or is_comment_block:
                i += 1
                continue
            
            # Collect msgstr block lines
            if line == 'msgstr ""':
                # Multi-line msgstr
                block_lines.append(lines[i])
                i += 1
                while i < len(lines) and lines[i].strip().startswith('"'):
                    block_lines.append(lines[i])
                    i += 1
                end_idx = i - 1
            else:
                # Single-line msgstr
                block_lines.append(lines[i])
                end_idx = i
                i += 1
            
            content = extract_content_from_lines(block_lines)
            
            # Store both escaped and unescaped versions
            escaped_content = ''.join([line.strip()[1:-1] if line.strip().startswith('"') and line.strip().endswith('"') 
                                     else line.split('"')[1].split('"')[0] if '"' in line 
                                     else '' for line in block_lines if '"' in line])
            
            with stats_lock:
                translation_stats['total_blocks'] += 1
            
            # Use enhanced content analysis
            should_translate, reason, detected_lang = should_translate_content(content, config.get('target_language', 'EN'))
            
            # Update statistics
            stats[reason] = stats.get(reason, 0) + 1
            if detected_lang != 'unknown':
                lang_stats[detected_lang] = lang_stats.get(detected_lang, 0) + 1
            
            # Update legacy stats for compatibility
            if reason in ['same_as_target']:
                update_stats('english')
            elif reason == 'empty':
                update_stats('empty')
            elif reason in ['different_language', 'unknown_assume_translate', 'coordinate']:
                update_stats('has_cyrillic')
            
            if should_translate:
                blocks.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'content': content,
                    'escaped_content': escaped_content,  # Store escaped version
                    'original_lines': [line.rstrip('\n\r') for line in block_lines],
                    'is_multiline': line == 'msgstr ""',
                    'reason': reason,
                    'detected_lang': detected_lang
                })
            
            continue
        
        i += 1
    
    # Print enhanced statistics
    total_blocks = len(blocks) + sum(stats.values()) - len(blocks)  # All blocks found
    translatable = len(blocks)
    
    target_lang_name = LANGUAGE_CODES.get(config.get('target_language', 'EN'), config.get('target_language', 'EN'))
    print(f"📊 Content Analysis (Target: {target_lang_name}):")
    print(f"   Total blocks: {total_blocks}")
    print(f"   Will translate: {translatable}")
    print(f"   Will skip: {total_blocks - translatable}")
    
    if lang_stats:
        print(f"\n📈 Detected languages:")
        for lang, count in sorted(lang_stats.items(), key=lambda x: x[1], reverse=True):
            lang_name = LANGUAGE_CODES.get(lang, lang)
            print(f"   {lang_name}: {count}")
    
    print(f"\n📋 Skip reasons:")
    for reason, count in stats.items():
        if count > 0 and reason not in ['different_language', 'unknown_assume_translate']:
            print(f"   {reason.replace('_', ' ').title()}: {count}")
    
    return blocks

def translate_batch_optimized(batch_data, batch_id, progress_bar, retry_count=0):
    """Optimized batch translation with coordinate handling and escaping fixes"""
    if not batch_data:
        return {}
    
    with api_semaphore:
        time.sleep(config["api_delay"])
        
        # Separate coordinates from regular text
        coordinate_items = []
        text_items = []
        
        for item in batch_data:
            if item.get('reason') == 'coordinate' or is_coordinate_text(item['content']):
                coordinate_items.append(item)
            else:
                text_items.append(item)
        
        results = {}
        
        # Handle coordinates directly (no API call needed)
        for item in coordinate_items:
            # Coordinate translation, keep escaped format if it was escaped
            translated = translate_coordinate(item['content'])
            # Apply proper escaping for PO format
            results[item['block_idx']] = escape_po_string(translated)
        
        # Handle text items with API
        if text_items:
            # Build batch text more efficiently
            batch_parts = []
            for item in text_items:
                # Use unescaped content for translation
                batch_parts.append(f"ID_{item['block_idx']}: {item['content']}")
            
            batch_text = "\n---\n".join(batch_parts)
            
            try:
                resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"You are a Russian-to-{config['target_language_name']} translator for gaming content. "
                                "CRITICAL RULES:\n"
                                "1. Keep ALL variables EXACTLY as-is: %(goal)s, %d, {0}, etc.\n"
                                "2. Keep ALL formatting EXACTLY as-is: \\n, \\t, line breaks, etc.\n"
                                "3. Keep ALL special characters and symbols EXACTLY as-is\n"
                                f"4. Translate Russian text to natural {config['target_language_name']}\n"
                                "5. Use gaming terminology (armor, destroy, battle, tank, etc.)\n"
                                "6. Keep ID_X format and separate with ---\n"
                                "7. PRESERVE exact spacing and formatting in the output\n"
                                "8. If text contains \\n characters, keep them EXACTLY as \\n\n\n"
                                "Format: ID_X: [exact translation with preserved formatting]"
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Translate to {config['target_language_name']} keeping ALL formatting:\n\n{batch_text}"
                        }
                    ],
                    temperature=0.1,
                    max_tokens=config["max_tokens"]
                )
                
                result = resp.choices[0].message.content.strip()
                
                # Parse results
                parts = result.split("---") if "---" in result else result.split("\n")
                
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    
                    match = re.match(r'ID_(\d+):\s*(.*)', part, re.DOTALL)
                    if match:
                        block_idx = int(match.group(1))
                        translation = match.group(2).strip()
                        
                        # Check for contamination (paths that shouldn't be there)
                        original = text_items[block_idx]['content'] if block_idx < len(text_items) else ""
                        if ('buyingPanel/' in translation or 'infoPanel/' in translation) and not ('buyingPanel/' in original or 'infoPanel/' in original):
                            print(f"⚠️  Detected contaminated translation, using original")
                            results[block_idx] = text_items[block_idx].get('escaped_content', escape_po_string(original))
                        else:
                            # Apply proper escaping for PO format
                            results[block_idx] = escape_po_string(translation)
                
                # Fallback for missing translations
                for i, item in enumerate(text_items):
                    if item['block_idx'] not in results:
                        # Use escaped version if available, otherwise escape the content
                        results[item['block_idx']] = item.get('escaped_content', escape_po_string(item['content']))
                
            except Exception as e:
                if retry_count < config["max_retries"]:
                    time.sleep(1 * (retry_count + 1))
                    return translate_batch_optimized(batch_data, batch_id, progress_bar, retry_count + 1)
                else:
                    print(f"❌ Batch {batch_id} failed: {e}")
                    # Fallback to originals with proper escaping
                    for item in text_items:
                        results[item['block_idx']] = item.get('escaped_content', escape_po_string(item['content']))
        
        with progress_lock:
            progress_bar.update(len(batch_data))
        
        return results

def create_new_msgstr_lines(original_block, new_content):
    """Create new msgstr lines preserving formatting and handling newlines properly"""
    original_lines = original_block['original_lines']
    first_line = original_lines[0]
    
    # Get indentation from original
    indent_match = re.match(r'^(\s*)', first_line)
    indent = indent_match.group(1) if indent_match else ""
    
    new_lines = []
    
    # Check if content has newlines or if original was multiline
    has_newlines = '\n' in new_content
    was_multiline = original_block['is_multiline']
    
    if has_newlines or was_multiline:
        # Handle multiline format
        new_lines.append(f'{indent}msgstr ""')
        
        if has_newlines:
            # Split by newlines and handle each part
            parts = new_content.split('\n')
            for i, part in enumerate(parts):
                # Content is already escaped, don't double-escape
                if i == len(parts) - 1:
                    # Last part - no \n suffix
                    new_lines.append(f'{indent}"{part}"')
                else:
                    # Add \n suffix for continuation
                    new_lines.append(f'{indent}"{part}\\n"')
        else:
            # No newlines but was originally multiline - keep as single quoted line
            # Content is already escaped
            new_lines.append(f'{indent}"{new_content}"')
    else:
        # Single line format
        # Content is already escaped
        new_lines.append(f'{indent}msgstr "{new_content}"')
    
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
    for i in range(0, len(blocks_to_translate), config["batch_size"]):
        batch_blocks = blocks_to_translate[i:i+config["batch_size"]]
        batch_data = []
        for block in batch_blocks:
            batch_data.append({
                'block_idx': block['block_idx'],
                'content': block['content'],
                'escaped_content': block.get('escaped_content', ''),
                'reason': block.get('reason', 'unknown')
            })
        batches.append((i // config["batch_size"], batch_data))
    
    print(f"🚀 Processing {len(batches)} batches with {config['max_workers']} workers...")
    
    progress_bar = tqdm(total=len(blocks_to_translate), 
                       desc=f"Translating {os.path.basename(input_path)}", 
                       leave=False)
    
    translations = {}
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
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
                    translations[item['block_idx']] = item.get('escaped_content', escape_po_string(item['content']))
    
    progress_bar.close()
    
    # Apply translations to create new file
    new_lines = []
    current_pos = 0
    
    sorted_blocks = sorted(blocks_to_translate, key=lambda x: x['start_idx'])
    
    for block in sorted_blocks:
        start_idx = block['start_idx']
        end_idx = block['end_idx']
        block_idx = block['block_idx']
        
        # Add lines before this block (preserving original exactly)
        new_lines.extend(lines[current_pos:start_idx])
        
        # Get translation (already properly escaped)
        translation = translations.get(block_idx, block.get('escaped_content', escape_po_string(block['content'])))
        
        # Create new msgstr lines with proper formatting
        new_msgstr_lines = create_new_msgstr_lines(block, translation)
        new_lines.extend([line + '\n' for line in new_msgstr_lines])
        
        current_pos = end_idx + 1
    
    # Add remaining lines (preserving original exactly)
    new_lines.extend(lines[current_pos:])
    
    # Write output with same encoding as input
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    
    return len(blocks_to_translate)

def main():
    """Main execution function"""
    global config, api_semaphore, client
    
    print("🚀 PO TRANSLATOR v12.0 - Fixed Double Escaping & Enhanced Detection")
    print("=" * 80)
    
    # Create directories
    created_dirs = create_directories()
    if created_dirs:
        print(f"📁 Created directories: {', '.join(created_dirs)}")
    
    # Load or create configuration (without language settings)
    config = load_config()
    if config is None:
        config = initial_setup()
    else:
        print(f"⚡ Settings loaded: {config['max_workers']} workers, batch size {config['batch_size']}")
    
    # ALWAYS prompt for language selection (not saved in config)
    lang_code, lang_name = setup_language()
    config["target_language"] = lang_code
    config["target_language_name"] = lang_name
    print(f"✅ Target language selected: {lang_name} ({lang_code})")
    print(f"🧠 Smart translation: Only translates text that's NOT already in {lang_name}")
    
    # Initialize semaphore with config
    api_semaphore = threading.Semaphore(config["max_workers"])
    
    # Check API key
    api_key = check_api_key()
    if not api_key:
        print("\n❌ Cannot proceed without API key. Exiting...")
        input("Press Enter to exit...")
        return
    
    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize OpenAI client: {e}")
        input("Press Enter to exit...")
        return
    
    # Check for PO files
    input_dir = "input"
    output_dir = "output"
    
    po_files = [f for f in os.listdir(input_dir) if f.endswith(".po")]
    
    if not po_files:
        print(f"\n📂 No .po files found in '{input_dir}' folder.")
        print("Please place your .po files there and run the program again.")
        input("Press Enter to exit...")
        return
    
    print(f"\n📋 Found {len(po_files)} .po files:")
    for fname in po_files:
        file_size = os.path.getsize(os.path.join(input_dir, fname)) / (1024*1024)
        print(f"  - {fname} ({file_size:.1f} MB)")
    
    # Confirm processing
    print(f"\n🎯 Ready to translate to {config['target_language_name']} ({config['target_language']})")
    proceed = input("Proceed with translation? (Y/n): ").strip().lower()
    if proceed == 'n':
        print("Translation cancelled.")
        return
    
    print("\n🚀 Starting translation process...")
    print("=" * 40)
    
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
    print(f"📊 Files processed: {len(po_files)}")
    print(f"📊 Strings translated: {total_translated}")
    print(f"🌍 Target language: {config['target_language_name']} ({config['target_language']})")
    print(f"⏱️ Total time: {total_time:.1f}s")
    if total_time > 0:
        print(f"📈 Average speed: {total_translated / total_time:.1f} strings/second")
    print(f"📁 Output folder: '{output_dir}'")
    print(f"⚙️ Configuration: Edit '{CONFIG_FILE}' to modify settings")
    
    # Show summary statistics
    if total_translated > 0:
        print(f"\n📋 Session Summary:")
        print(f"   • Batch size: {config['batch_size']} strings per API call")
        print(f"   • Workers: {config['max_workers']} parallel threads")
        print(f"   • API delay: {config['api_delay']}s between calls")
        estimated_api_calls = (total_translated + config['batch_size'] - 1) // config['batch_size']
        print(f"   • Estimated API calls made: ~{estimated_api_calls}")
    
    print(f"\n✨ Thank you for using PO Translator!")
    print("💡 Tip: You can run the program again to process new files.")
    print("    Language will be prompted each time, other settings are saved.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Translation interrupted by user")
        print("⚠️  Partial results may be available in the output folder.")
        print("💡 Tip: You can resume by running the program again.")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"\n❌ Unexpected error occurred:")
        print(f"   {str(e)}")
        print("\n🔧 Troubleshooting suggestions:")
        print("   1. Check your API key in 'key.txt'")
        print("   2. Verify your internet connection")
        print("   3. Make sure .po files are valid")
        print("   4. Try deleting 'config.json' to reset settings")
        print(f"   5. Check that '{CONFIG_FILE}' is not corrupted")
        input("\nPress Enter to exit...")