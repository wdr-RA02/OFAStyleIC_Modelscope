#!/bin/bash

# 定义命令行参数的名称、默认值、是否必须等信息
input_file=""
output_file="output.txt"
help=false

# 使用 getopts 命令解析命令行参数
while getopts "f:o:h" opt; do
  case $opt in
    f)
      input_file="$OPTARG"
      ;;
    o)
      output_file="$OPTARG"
      ;;
    h)
      help=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument" >&2
      exit 1
      ;;
  esac
done

# 对解析后的参数进行检查和处理
if $help; then
  echo "Usage: $0 [-f input_file] [-o output_file] [-h]"
  echo "  -f input_file: path to the input file (required)"
  echo "  -o output_file: path to the output file (default: output.txt)"
  echo "  -h: display this help message"
  exit 0
fi

if [ -z "$input_file" ]; then
  echo "Error: input file not specified" >&2
  exit 1
fi

# 进行操作
echo "Input file: $input_file"
echo "Output file: $output_file"
