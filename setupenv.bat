
REM Install core dependencies
echo ✅ Installing core dependencies...
pip install pandas>=1.3.0
pip install numpy>=1.20.0
pip install openai>=0.27.0
pip install transformers>=4.15.0
pip install scikit-learn>=1.0.0
pip install tqdm>=4.62.0
pip install requests>=2.25.0
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0

REM Install CPU-only PyTorch
echo ✅ Installing CPU-only PyTorch...
pip install torch>=1.10.0 --index-url https://download.pytorch.org/whl/cpu

REM Clone TruthfulQA repository
if not exist "TruthfulQA" (
    echo ✅ Cloning TruthfulQA repository...
    git clone https://github.com/sylinrl/TruthfulQA.git
    if %errorlevel% neq 0 (
        echo ❌ Failed to clone TruthfulQA repository
        echo    Make sure git is installed and accessible
        pause
        exit /b 1
    )
) else (
    echo ✅ TruthfulQA repository already exists
)

REM Test the setup
echo ✅ Testing the setup...
python -c "import sys; sys.path.insert(0, 'TruthfulQA'); from truthfulqa.metrics import run_gpt_judge_forwarding; from truthfulqa.utilities import split_multi_answer; print('✅ Successfully imported TruthfulQA functions!')"
if %errorlevel% neq 0 (
    echo ❌ Import test failed
    pause
    exit /b 1
)

echo.
echo 🎉 Environment setup completed successfully!
echo.
echo 🎯 Next steps:
echo    1. Activate the environment: conda activate %ENV_NAME%
echo    2. Set your OpenAI API key: set OPENAI_API_KEY=your-key-here
echo    3. Run evaluation: python official_tqa_eval.py your_predictions.json
echo.
echo 📋 Environment details:
echo    - Name: %ENV_NAME%
echo    - Python: 3.9
echo    - TruthfulQA: .\TruthfulQA\
echo    - Dependencies: minimal set (no TensorFlow/T5)
echo.
pause