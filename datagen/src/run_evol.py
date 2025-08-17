#!/usr/bin/env python3
"""
Simple runner script for zebra puzzle evolution with minimal dependencies.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")  # Load from parent directory

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please set these in your .env file or environment.")
        return False
    
    print("✅ Environment variables OK")
    return True

def main():
    """Run the evolution pipeline."""
    print("🧩 Zebra Puzzle Evolution Runner")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run tests first
    print("\n📋 Running tests...")
    try:
        from test_evol import main as test_main
        if not test_main():
            print("❌ Tests failed. Please fix issues before running evolution.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test import failed: {e}")
        sys.exit(1)
    
    # Run evolution
    print("\n🚀 Starting evolution pipeline...")
    try:
        from evol_zebra_puzzles import main as evol_main
        evol_main()
        print("\n🎉 Evolution completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Evolution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evolution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()