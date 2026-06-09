# 00_check_columns.py
# -*- coding: utf-8 -*-

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 CSV 파일 경로")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print("\n========== 컬럼 목록 ==========")
    for i, col in enumerate(df.columns, start=1):
        print(f"{i}. {col}")

    print("\n========== 데이터 크기 ==========")
    print(df.shape)

    print("\n========== 앞 5행 ==========")
    print(df.head())

    print("\n========== 결측치 개수 ==========")
    print(df.isna().sum())


if __name__ == "__main__":
    main()