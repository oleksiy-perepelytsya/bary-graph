"""Sample N random MetaBary docs (L10-L13) per level via MCP and write to JSONL.

Usage:
    python scripts/sample_by_level.py [n_per_level] [output_path]

Defaults: n=250, output=data/sample_by_level.jsonl
"""
import asyncio
import json
import pathlib
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
OUTPUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "data/sample_by_level.jsonl")
LEVELS = [10, 11, 12, 13]

SERVER = StdioServerParameters(
    command="python",
    args=["-m", "scripts.mcp_server"],
)


async def main() -> None:
    OUTPUT.parent.mkdir(exist_ok=True)
    total = 0

    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            with OUTPUT.open("w") as fh:
                for level in LEVELS:
                    result = await session.call_tool(
                        "sample_metabary",
                        {"level": level, "n": N, "with_parent": False},
                    )
                    raw = result.content[0].text if result.content else "[]"
                    records = json.loads(raw)
                    for rec in records:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count = len(records)
                    total += count
                    print(f"  level {level}: {count} docs", file=sys.stderr)

    print(f"Wrote {total} records → {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
