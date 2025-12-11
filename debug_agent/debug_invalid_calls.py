"""
Debug script to capture and log raw invalid_tool_calls from Qwen.

This script will:
1. Create a minimal agent
2. Log all invalid_tool_calls before any fixing
3. Save patterns to a file for analysis
"""

import json
import logging
from pathlib import Path

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from langchain_qwq import ChatQwen
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File to save invalid calls
INVALID_CALLS_LOG = Path(__file__).parent / "invalid_calls_log.jsonl"


@tool
def test_tool(bounds: list) -> str:
    """Test tool that expects a list of bounds.

    Args:
        bounds: List of [min, max] pairs for each variable
    """
    return f"Received {len(bounds)} bounds"


def test_invalid_calls():
    """Test what invalid_tool_calls look like from Qwen."""

    print("=" * 70)
    print("DEBUGGING INVALID_TOOL_CALLS FROM QWEN")
    print("=" * 70)

    # Initialize Qwen
    llm = ChatQwen(model="qwen-flash", temperature=0.0)
    llm_with_tools = llm.bind_tools([test_tool])

    # Test prompts that likely trigger Python syntax
    test_cases = [
        "Use test_tool with 10 bounds, each being [-5, 10]",
        "Call test_tool with bounds parameter set to 5 copies of [-5, 10]",
        "Invoke test_tool with bounds=list of 3 items, each [-5, 10]",
        "Call test_tool with bounds that repeats [-5, 10] ten times",
        "Use test_tool where bounds=[[-5,10]] repeated 15 times",
        "Call test_tool, bounds should be [-5, 10] duplicated 7 times",
    ]

    all_invalid_calls = []

    for i, prompt in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"Test Case {i+1}: {prompt}")
        print('='*70)

        messages = [HumanMessage(content=prompt)]

        try:
            response = llm_with_tools.invoke(messages)

            # Log tool_calls (valid)
            tool_calls = getattr(response, 'tool_calls', [])
            print(f"\nValid tool_calls: {len(tool_calls)}")
            for tc in tool_calls:
                print(f"  - {tc['name']}: {tc['args']}")

            # Log invalid_tool_calls (invalid)
            invalid_calls = getattr(response, 'invalid_tool_calls', [])
            print(f"\nInvalid tool_calls: {len(invalid_calls)}")

            if invalid_calls:
                for inv in invalid_calls:
                    print(f"\n  Invalid Call #{len(all_invalid_calls) + 1}:")
                    print(f"    Name: {inv.get('name', 'UNKNOWN')}")
                    print(f"    ID: {inv.get('id', 'UNKNOWN')}")
                    print(f"    Error: {inv.get('error', 'UNKNOWN')}")
                    print(f"    Args (raw): {inv.get('args', 'UNKNOWN')}")
                    print(f"    Type: {type(inv.get('args', ''))}")

                    # Save to log
                    all_invalid_calls.append({
                        "test_case": i + 1,
                        "prompt": prompt,
                        "invalid_call": {
                            "name": inv.get('name'),
                            "id": inv.get('id'),
                            "error": inv.get('error'),
                            "args": inv.get('args'),
                            "type": str(type(inv.get('args')))
                        }
                    })
            else:
                print("  (No invalid calls)")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save all invalid calls to file
    if all_invalid_calls:
        print(f"\n{'='*70}")
        print(f"Saving {len(all_invalid_calls)} invalid calls to: {INVALID_CALLS_LOG}")
        print('='*70)

        with open(INVALID_CALLS_LOG, 'w') as f:
            for entry in all_invalid_calls:
                f.write(json.dumps(entry) + '\n')

        print(f"Log saved!")
    else:
        print(f"\n{'='*70}")
        print("No invalid calls captured! Qwen might have generated valid JSON.")
        print('='*70)

    return all_invalid_calls


if __name__ == "__main__":
    results = test_invalid_calls()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Total invalid calls captured: {len(results)}")

    if results:
        print("\nPatterns observed:")
        for i, entry in enumerate(results):
            args = entry['invalid_call']['args']
            print(f"\n{i+1}. Args string: {args[:100]}...")
