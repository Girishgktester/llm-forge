import sys
sys.argv = ["langgraph", "dev", "--port", "2024"]
from langgraph_cli.cli import cli
cli()
