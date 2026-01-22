#!/bin/bash

# Setup script for Coalition Formation Visualizer
# Run this script to install all dependencies and verify the installation

set -e  # Exit on error

echo "========================================"
echo "Coalition Formation Visualizer Setup"
echo "========================================"
echo ""

# Check Python
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Error: Python not found. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"
echo ""

# Check Node.js
echo "Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js not found. Please install Node.js 18+ first."
    echo "   Download from: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version)
echo "✓ Found Node.js $NODE_VERSION"
echo ""

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm not found. Please install npm first."
    exit 1
fi

NPM_VERSION=$(npm --version)
echo "✓ Found npm $NPM_VERSION"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
echo "Running: $PYTHON_CMD -m pip install -r requirements.txt"
$PYTHON_CMD -m pip install -r requirements.txt || {
    echo "❌ Failed to install Python dependencies"
    exit 1
}
echo "✓ Python dependencies installed"
echo ""

# Install Node.js dependencies
echo "Installing Node.js dependencies for visualizer..."
cd viz
echo "Running: npm install"
npm install || {
    echo "❌ Failed to install Node.js dependencies"
    exit 1
}
cd ..
echo "✓ Node.js dependencies installed"
echo ""

# Verify installation
echo "Verifying installation..."

# Check if viz_service can be imported
$PYTHON_CMD -c "import viz_service; print('✓ viz_service module can be imported')" || {
    echo "❌ Failed to import viz_service"
    exit 1
}

# Check if strategy tables exist
if [ ! -d "strategy_tables" ]; then
    echo "⚠️  Warning: strategy_tables directory not found"
else
    XLSX_COUNT=$(find strategy_tables -name "*.xlsx" ! -name ".~lock*" | wc -l)
    echo "✓ Found $XLSX_COUNT XLSX files in strategy_tables/"
fi

echo ""
echo "========================================"
echo "✓ Installation Complete!"
echo "========================================"
echo ""
echo "To start the visualizer:"
echo ""
echo "1. Start the backend (in this terminal):"
echo "   $PYTHON_CMD -m viz_service"
echo ""
echo "2. Start the frontend (in a new terminal):"
echo "   cd viz"
echo "   npm run dev"
echo ""
echo "3. Open your browser to: http://localhost:3000"
echo ""
echo "For more information, see:"
echo "  - VIZ_README.md (quick start)"
echo "  - viz/README.md (detailed docs)"
echo "  - VISUALIZER_IMPLEMENTATION.md (technical details)"
echo ""
