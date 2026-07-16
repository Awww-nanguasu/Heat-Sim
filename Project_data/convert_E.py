# from pathlib import Path
# import time

# import numpy as np


# INPUT_FILE = Path(r"C:\git-workplace\HeatSim\Project_data\E.txt")
# OUTPUT_FILE = Path(r"C:\git-workplace\HeatSim\Project_data\E_diag.npy")

# PROGRESS_EVERY = 10_000_000


# def main() -> None:
#     start_time = time.time()

#     with INPUT_FILE.open("r", encoding="utf-8") as file:
#         header = file.readline().split()

#         if len(header) != 2:
#             raise ValueError(f"文件头格式错误：{header}")

#         nrow, ncol = map(int, header)

#         if nrow != ncol:
#             raise ValueError(f"E 不是方阵：{nrow} × {ncol}")

#         print(f"E shape: {nrow:,} × {ncol:,}")
#         print(f"预计输出大小: {nrow * 8 / 1024**3:.3f} GiB")

#         # 直接在内存中创建对角向量，约占 1.29 GiB
#         diagonal = np.empty(nrow, dtype=np.float64)

#         count = 0

#         for line_number, line in enumerate(file, start=2):
#             parts = line.split()

#             if not parts:
#                 continue

#             if len(parts) != 3:
#                 raise ValueError(
#                     f"第 {line_number} 行格式错误，应为 row col value：{line[:100]}"
#                 )

#             row = int(parts[0])
#             col = int(parts[1])
#             value = float(parts[2])

#             if row != col:
#                 raise ValueError(
#                     f"第 {line_number} 行不是对角元素："
#                     f"row={row}, col={col}, value={value}"
#                 )

#             if row < 0 or row >= nrow:
#                 raise IndexError(
#                     f"第 {line_number} 行索引越界：{row}"
#                 )

#             # 根据当前文件格式，索引应按 0, 1, 2, ... 连续出现
#             if row != count:
#                 raise ValueError(
#                     f"第 {line_number} 行索引不连续："
#                     f"期望 {count}，实际 {row}"
#                 )

#             diagonal[row] = value
#             count += 1

#             if count % PROGRESS_EVERY == 0:
#                 elapsed = time.time() - start_time
#                 rate = count / elapsed

#                 print(
#                     f"已转换 {count:,}/{nrow:,} "
#                     f"({count / nrow * 100:.2f}%)，"
#                     f"速度 {rate:,.0f} 行/秒"
#                 )

#     if count != nrow:
#         raise ValueError(
#             f"对角元素数量错误：期望 {nrow:,}，实际 {count:,}"
#         )

#     np.save(OUTPUT_FILE, diagonal)

#     elapsed = time.time() - start_time

#     print("=" * 60)
#     print("转换完成")
#     print(f"输出文件：{OUTPUT_FILE}")
#     print(f"shape：{diagonal.shape}")
#     print(f"dtype：{diagonal.dtype}")
#     print(f"元素数量：{count:,}")
#     print(f"最小值：{diagonal.min():.16e}")
#     print(f"最大值：{diagonal.max():.16e}")
#     print(f"前 10 项：{diagonal[:10]}")
#     print(f"总耗时：{elapsed:.1f} 秒")


# if __name__ == "__main__":
#     main()

from pathlib import Path

b_file = Path(r"C:\git-workplace\HeatSim\Project_data\B.txt")

target_col = 11022
count = 0
samples = []
value_sum = 0.0
min_row = None
max_row = None

with b_file.open("r", encoding="utf-8") as f:
    f.readline()  # 跳过矩阵尺寸

    for line in f:
        row_s, col_s, value_s = line.split()
        col = int(col_s)

        if col == target_col:
            row = int(row_s)
            value = float(value_s)

            count += 1
            value_sum += value
            min_row = row if min_row is None else min(min_row, row)
            max_row = row if max_row is None else max(max_row, row)

            if len(samples) < 20:
                samples.append((row, value))

print("最后一列非零元数量:", count)
print("行号范围:", min_row, max_row)
print("系数和:", value_sum)
print("前 20 个非零元:")
for item in samples:
    print(item)