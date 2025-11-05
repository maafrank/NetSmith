#!/bin/bash

# NetSmith Extension Publishing Script
# This script builds and publishes the extension to VS Code Marketplace

set -e  # Exit on error

echo "🔨 Building NetSmith extension..."

# Load environment variables
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with VSCE_PAT token"
    echo "See .env.example for template"
    exit 1
fi

source .env

if [ -z "$VSCE_PAT" ]; then
    echo "❌ Error: VSCE_PAT not set in .env file"
    exit 1
fi

# Build the extension
npm run build

echo "📦 Packaging extension..."

# Package (creates .vsix file)
vsce package

echo "🚀 Publishing to VS Code Marketplace..."

# Publish using token from environment
echo "$VSCE_PAT" | vsce publish -p "$VSCE_PAT"

echo "✅ Successfully published NetSmith!"
echo "View at: https://marketplace.visualstudio.com/items?itemName=MatthewFrank.netsmith"
