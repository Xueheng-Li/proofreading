# LaTeX Proofreader  文档校对工具 📚

This project provides a script to proofread LaTeX documents using a language model. The script processes LaTeX files, identifies sections to proofread, and uses the language model to improve the text. ✨


本项目提供了一个使用语言模型校对 LaTeX 文档的脚本。该脚本可以处理 LaTeX 文件，识别需要校对的部分，并使用语言模型改进文本内容。 ✨

## 项目初衷 🎯

Proofreading long LaTeX manuscripts can be a time-consuming and error-prone task for researchers. This project aims to automate the proofreading process, allowing researchers to focus on the content of their work rather than the tedious task of checking for grammatical errors, typos, and formatting issues. By leveraging a powerful language model, this script can efficiently **automate the proofreading of entire LaTeX documents by LLM in one go**, ensuring consistency and accuracy throughout the manuscript. This is particularly beneficial for researchers who need to submit high-quality papers to journals and conferences.

对于研究人员来说，校对长篇 LaTeX 手稿可能是一项耗时且容易出错的任务。本项目旨在自动化校对过程，使研究人员能够专注于他们的工作内容，而不是繁琐的语法错误、拼写错误和格式问题检查。通过利用强大的语言模型，该脚本可以**利用大语言模型全自动校对整个论文 LaTeX 文件**，确保手稿的一致性和准确性。这对需要向期刊和会议提交高质量论文的研究人员特别有益。

## Installation 🛠️

1. Install Python3.8+ if you haven't already.

2. Install the required package:
    ```sh
    pip install -qU langchain-openai
    ```
3. Clone this repository:
    ```sh
    git clone https://github.com/Xueheng-Li/proofreading.git
    cd proofreading
    ```

## Configuration ⚙️

1. Rename `setting.py.example` to `setting.py`:
    ```sh
    mv setting.py.example setting.py
    ```

2. Open `setting.py` and set your API key and base URL:
    ```python
    # setting.py
    API_KEY = "your_api_key"
    BASE_URL = "any_openai_compatible_provider_base_url" # e.g., "https://openrouter.ai/api/v1"
    MODEL_NAME = "model_name" # e.g., "gpt-4o"
    ```

## Usage 🚀

To run the proofreader script, use the following command: 💻

```sh
python proofread.py --input INPUT_FILE --output OUTPUT_FILE [--no-resume] [--stream]
```

Arguments:
- `--input`: Path to the input LaTeX file 📄
- `--output`: Path to the output LaTeX file 📝
- `--no-resume`: Start fresh, ignoring any saved progress 🔄
- `--stream`: Stream model output in real-time ⚡

### Example 📝

Basic usage:

```sh
python proofread.py --input path/to/your/manuscript.tex --output copy_edited.tex
```

This command will proofread `path/to/your/manuscript.tex` and save the results to `copy_edited.tex`, starting from scratch and streaming the model output in real-time.

## Logging 📊

The script generates a log file in the same directory as the output file. The log file contains detailed information about the proofreading process. 📋

## License 📜

This project is licensed under the MIT License. ⚖️

---

# 中文说明 📚


## 安装 🛠️

1. 如果尚未安装，请安装 Python3.8+。

2. 安装必需的包：
    ```sh
    pip install -qU langchain-openai
    ```

## 配置 ⚙️

1. 将 `setting.py.example` 重命名为 `setting.py`：
    ```sh
    mv setting.py.example setting.py
    ```

2. 打开 `setting.py` 并设置您的 API 密钥和基础 URL：
    ```python
    # setting.py
    API_KEY = "你的_api_密钥"
    BASE_URL = "任何_openai_兼容的_提供商基础_url" # 例如："https://openrouter.ai/api/v1"
    MODEL_NAME = "模型名称" # 例如："gpt-4"
    ```

## 使用方法 🚀

使用以下命令运行校对脚本：💻

```sh
python proofread.py --input 输入文件 --output 输出文件 [--no-resume] [--stream]
```

参数说明：
- `--input`：输入 LaTeX 文件的路径 📄
- `--output`：输出 LaTeX 文件的路径 📝
- `--no-resume`：从头开始，忽略任何已保存的进度 🔄
- `--stream`：实时流式输出模型结果 ⚡

### 示例 📝

基本用法：

```sh
python proofread.py --input path/to/your/manuscript.tex --output copy_edited.tex
```

此命令将校对 `path/to/your/manuscript.tex` 并将结果保存到 `copy_edited.tex`，从头开始并实时流式输出模型结果。

## 日志记录 📊

脚本会在输出文件的同一目录下生成日志文件。日志文件包含有关校对过程的详细信息。 📋

## 许可证 📜

本项目基于 MIT 许可证开源。 ⚖️
