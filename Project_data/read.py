# from pathlib import Path
# import os


# def inspect_sparse_txt(path, preview_lines=10, block_size=1024 * 1024 * 256):
#     path = Path(path)

#     print("=" * 80)
#     print("file:", path)
#     print("file size GB:", path.stat().st_size / 1024**3)

#     # 1. 读取前几行，看格式
#     print("\nFirst lines:")
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         lines = []
#         for i in range(preview_lines):
#             line = f.readline()
#             if not line:
#                 break
#             lines.append(line.rstrip("\n"))
#             print(f"{i + 1}: {repr(line.rstrip())}")

#     # 2. 尝试解析第一行矩阵尺寸
#     first = lines[0].split()
#     if len(first) == 2:
#         try:
#             nrow, ncol = map(int, first)
#             print("\nDetected possible matrix shape:", (nrow, ncol))
#         except ValueError:
#             nrow, ncol = None, None
#             print("\nFirst line has 2 columns, but not integer shape.")
#     else:
#         nrow, ncol = None, None
#         print("\nFirst line does not look like shape line.")

#     # 3. 检查后面几行列数
#     print("\nColumn counts in preview:")
#     for i, line in enumerate(lines):
#         print(i + 1, len(line.split()))

#     # 4. 快速统计行数，估算 nnz
#     print("\nCounting lines, this scans the file once but does not parse numbers...")
#     line_count = 0
#     with open(path, "rb") as f:
#         while True:
#             block = f.read(block_size)
#             if not block:
#                 break
#             line_count += block.count(b"\n")

#     # 如果文件最后没有换行，补一行
#     with open(path, "rb") as f:
#         f.seek(0, os.SEEK_END)
#         if f.tell() > 0:
#             f.seek(-1, os.SEEK_END)
#             last_char = f.read(1)
#             if last_char != b"\n":
#                 line_count += 1

#     print("total lines:", line_count)

#     if len(first) == 2:
#         nnz = line_count - 1
#         print("estimated nnz:", nnz)
#     else:
#         nnz = line_count
#         print("estimated nnz, if no header:", nnz)

#     # 5. 估算稀疏度
#     if nrow is not None and ncol is not None:
#         density = nnz / (nrow * ncol)
#         print("density:", density)
#         print("density percent:", density * 100)

#     # 6. 估算转成 CSR 后的大小
#     # CSR 大概需要:
#     # data: float64, 8 bytes * nnz
#     # indices: int32, 4 bytes * nnz
#     # indptr: int32/int64, about 4 or 8 bytes * (nrow + 1)
#     csr_float64_int32_gb = (8 * nnz + 4 * nnz + 4 * ((nrow or 0) + 1)) / 1024**3
#     csr_float64_int64_gb = (8 * nnz + 8 * nnz + 8 * ((nrow or 0) + 1)) / 1024**3

#     print("\nEstimated CSR binary size:")
#     print("float64 data + int32 index GB:", csr_float64_int32_gb)
#     print("float64 data + int64 index GB:", csr_float64_int64_gb)

#     print("=" * 80)


# inspect_sparse_txt(r"C:\git-workplace\HeatSim\Project_data\A.txt")

from pathlib import Path
import os


def inspect_sparse_txt(path, preview_lines=10, block_size=1024 * 1024 * 256):
    path = Path(path)

    print("=" * 100)
    print("file:", path.name)
    print("file size GB:", path.stat().st_size / 1024**3)

    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i in range(preview_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))

    print("\nFirst lines:")
    for i, line in enumerate(lines):
        print(f"{i + 1}: {repr(line)}")

    print("\nColumn counts:")
    for i, line in enumerate(lines):
        print(f"{i + 1}: {len(line.split())}")

    first = lines[0].split()

    nrow = None
    ncol = None
    has_header = False

    if len(first) == 2:
        try:
            nrow, ncol = map(int, first)
            has_header = True
            print("\nDetected matrix shape:", (nrow, ncol))
        except ValueError:
            print("\nFirst line has 2 columns but is not integer shape.")
    else:
        print("\nFirst line does not look like a shape header.")

    print("\nCounting lines...")
    line_count = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            line_count += block.count(b"\n")

    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                line_count += 1

    if has_header:
        nnz = line_count - 1
    else:
        nnz = line_count

    print("total lines:", line_count)
    print("estimated nnz:", nnz)

    if nrow is not None and ncol is not None:
        density = nnz / (nrow * ncol)
        print("density:", density)
        print("density percent:", density * 100)

        csr_float64_int32_gb = (8 * nnz + 4 * nnz + 4 * (nrow + 1)) / 1024**3
        csr_float64_int64_gb = (8 * nnz + 8 * nnz + 8 * (nrow + 1)) / 1024**3

        print("\nEstimated CSR binary size:")
        print("float64 data + int32 index GB:", csr_float64_int32_gb)
        print("float64 data + int64 index GB:", csr_float64_int64_gb)

    if len(lines) >= 2:
        second = lines[1].split()
        if len(second) == 3:
            try:
                r = int(second[0])
                c = int(second[1])
                v = float(second[2])
                print("\nTriplet sample parsed:")
                print("row:", r, "col:", c, "value:", v)

                if r == 0 or c == 0:
                    print("indexing: likely 0-based")
                elif r == 1 or c == 1:
                    print("indexing: possibly 1-based")
            except ValueError:
                print("\nSecond line has 3 columns but cannot parse as row col value.")

    print("=" * 100)
    print()


base = Path(r"C:\git-workplace\HeatSim\Project_data")

for name in ["B.txt", "E.txt"]:
    inspect_sparse_txt(base / name)