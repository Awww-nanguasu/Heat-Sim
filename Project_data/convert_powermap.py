from pathlib import Path
import csv
import re

import numpy as np


input_file = Path(
    r"C:\git-workplace\HeatSim\Project_data\powermap.txt"
)
output_file = Path(
    r"C:\git-workplace\HeatSim\Project_data\u.npy"
)

N_INPUTS = 11023
BD_INDEX = 11022

u = np.zeros(N_INPUTS, dtype=np.float64)
seen = np.zeros(N_INPUTS, dtype=bool)

with input_file.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as file:
    reader = csv.DictReader(file)

    if reader.fieldnames != ["PowerName", "Value"]:
        raise ValueError(
            f"文件头不符合预期：{reader.fieldnames}"
        )

    for line_number, row in enumerate(reader, start=2):
        name = row["PowerName"].strip()
        value = float(row["Value"])

        match = re.fullmatch(r"source(\d+)", name)

        if match:
            source_number = int(match.group(1))
            index = source_number - 1
        elif name == "bd-0":
            index = BD_INDEX
        else:
            raise ValueError(
                f"第 {line_number} 行名称无法识别：{name}"
            )

        if not 0 <= index < N_INPUTS:
            raise IndexError(
                f"第 {line_number} 行索引越界："
                f"{name} -> {index}"
            )

        if seen[index]:
            raise ValueError(
                f"第 {line_number} 行存在重复输入：{name}"
            )

        u[index] = value
        seen[index] = True

missing = np.flatnonzero(~seen)

if len(missing) != 0:
    raise ValueError(
        f"缺少 {len(missing)} 个输入，"
        f"前几个缺失索引：{missing[:20]}"
    )

np.save(output_file, u)

print("转换完成")
print("输出文件：", output_file)
print("shape：", u.shape)
print("dtype：", u.dtype)
print("source1：", u[0])
print("source11022：", u[11021])
print("bd-0：", u[11022])
print("功率源之和：", u[:11022].sum())