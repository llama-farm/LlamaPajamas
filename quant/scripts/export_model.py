#!/usr/bin/env python3
"""Wrapper script for unified model export."""
import sys
from llama_pajamas_quant.exporters.unified import main

if __name__ == '__main__':
    sys.exit(main())
