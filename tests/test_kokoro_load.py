import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

try:
    from kokoro import KPipeline
    print("✅ Successfully imported KPipeline")
    
    # Try initializing with just lang_code
    try:
        pipeline = KPipeline(lang_code='a')
        print("✅ Successfully initialized KPipeline(lang_code='a')")
    except TypeError as e:
        print(f"❌ Failed to initialize KPipeline(lang_code='a'): {e}")
except ImportError:
    print("❌ Kokoro library not found. Cannot verify locally.")
