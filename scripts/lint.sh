#!/bin/bash
# Lint and format all files using pre-commit

echo "🔍 Running pre-commit on all files..."
uv run pre-commit run --all-files

echo ""
echo "✅ Pre-commit completed!"
echo ""
echo "💡 To install pre-commit hooks (run automatically on git commit):"
echo "   uv run pre-commit install"
echo ""
echo "💡 To run pre-commit on specific files:"
echo "   uv run pre-commit run --files <file1> <file2>"
