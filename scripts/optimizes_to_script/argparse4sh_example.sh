#!/bin/bash
source argparse4sh.sh

# 定义命令行参数
argparse4sh_setup "example script" \
    "This is an example script demonstrating argparse4sh" \
    "" \
    "author=John Doe" \
    "version=1.0" \
    "" \
    "-f|--file=FILE: path to the input file (required)" \
    "-o|--output=FILE: path to the output file (default: output.txt)" \
    "-h|--help: display this help message"

# 解析命令行参数
argparse4sh_parse "$@"

# 获取解析后的参数
input_file=$(argparse4sh_getarg file)
output_file=$(argparse4sh_getarg output "output.txt")

# 检查参数
if [ -z "$input_file" ]; then
    echo "Error: input file not specified" >&2
    exit 1
fi

# 进行操作
echo "Input file: $input_file"
echo "Output file: $output_file"
