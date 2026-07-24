"""Backward-compatible launcher for the aegis-mcp server.

The implementation now lives in :mod:`monai_aegis.mcp_server`, installed with
the package and runnable as the ``aegis-mcp`` console command
(``pip install 'monai-aegis[mcp]'``). This root script is retained so
``python aegis_mcp_server.py`` and existing MCP client configs that point at
this path keep working unchanged.
"""
from monai_aegis.mcp_server import main

if __name__ == "__main__":
    main()
