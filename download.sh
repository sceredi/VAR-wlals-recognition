#!/usr/bin/env bash

FILE_ID="1QbuUJbwrq0D3hU8-sEePb4tJ87t2WA8r"
OUTPUT_NAME="wlasl.zip"

if ! command -v gdown &> /dev/null
then
    echo "gdown not found! You can install it by running: pip install gdown"
    exit 1
fi

gdown "${FILE_ID}" -O "${OUTPUT_NAME}"

if [ -f "${OUTPUT_NAME}" ]; then
    echo "Download completed: ${OUTPUT_NAME}"
else
    echo "Error downloading file!"
    exit 1
fi

unzip "${OUTPUT_NAME}"

