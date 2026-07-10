from pathlib import Path
import json
import time

import numpy as np
from scipy.sparse import coo_matrix


BASE_DIR = Path(r"C:\git-workplace\HeatSim\Project_data")

B_TXT = BASE_DIR / "B.txt"
U_NPY = BASE_DIR / "u.npy"

B_OUTPUT_DIR = BASE_DIR / "B_csr"
RHS_OUTPUT = BASE_DIR / "rhs.npy"


def main() -> None:
    start_time = time.time()

    if not B_TXT.is_file():
        raise FileNotFoundError(f"找不到 B.txt：{B_TXT}")

    if not U_NPY.is_file():
        raise FileNotFoundError(f"找不到 u.npy：{U_NPY}")

    B_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 读取矩阵尺寸
    with B_TXT.open("r", encoding="utf-8") as file:
        header = file.readline().split()

    if len(header) != 2:
        raise ValueError(f"B.txt 文件头错误：{header}")

    nrow, ncol = map(int, header)

    print(f"B shape: {nrow:,} × {ncol:,}")
    print("正在读取 B.txt ...")

    # 直接按整数、浮点类型读取，跳过第一行矩阵尺寸
    dtype = np.dtype([
        ("row", np.int64),
        ("col", np.int32),
        ("value", np.float64),
    ])

    triplets = np.loadtxt(
        B_TXT,
        skiprows=1,
        dtype=dtype,
    )

    rows = triplets["row"]
    cols = triplets["col"]
    values = triplets["value"]

    nnz_input = len(values)

    print(f"读取非零元数量：{nnz_input:,}")
    print(f"row 范围：{rows.min():,} ～ {rows.max():,}")
    print(f"col 范围：{cols.min():,} ～ {cols.max():,}")

    if rows.min() < 0 or rows.max() >= nrow:
        raise IndexError("B.txt 中存在越界的行号")

    if cols.min() < 0 or cols.max() >= ncol:
        raise IndexError("B.txt 中存在越界的列号")

    print("正在构造 CSR 矩阵 ...")

    B = coo_matrix(
        (values, (rows, cols)),
        shape=(nrow, ncol),
        dtype=np.float64,
    ).tocsr()

    # 如果原始 COO 有重复坐标，将重复值合并
    B.sum_duplicates()
    B.sort_indices()

    print("CSR 构造完成")
    print(f"CSR shape：{B.shape}")
    print(f"CSR nnz：{B.nnz:,}")
    print(f"indptr dtype：{B.indptr.dtype}")
    print(f"indices dtype：{B.indices.dtype}")
    print(f"data dtype：{B.data.dtype}")

    # 保存为和 A 相似的三个数组
    print("正在保存 B CSR 数据 ...")

    np.save(B_OUTPUT_DIR / "indptr.npy", B.indptr)
    np.save(B_OUTPUT_DIR / "indices.npy", B.indices)
    np.save(B_OUTPUT_DIR / "data.npy", B.data)

    meta = {
        "shape": [int(nrow), int(ncol)],
        "nnz": int(B.nnz),
        "format": "csr",
        "data_dtype": str(B.data.dtype),
        "indices_dtype": str(B.indices.dtype),
        "indptr_dtype": str(B.indptr.dtype),
        "index_base": 0,
    }

    with (B_OUTPUT_DIR / "meta.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(meta, file, indent=2)

    # 读取输入向量 u
    u = np.load(U_NPY)

    print(f"u shape：{u.shape}")
    print(f"u dtype：{u.dtype}")
    print(f"u 最后一项：{u[-1]}")

    if u.ndim != 1:
        raise ValueError(f"u 必须是一维向量，当前 shape={u.shape}")

    if len(u) != ncol:
        raise ValueError(
            f"维度不匹配：B 有 {ncol} 列，但 u 长度为 {len(u)}"
        )

    if not np.isfinite(u).all():
        raise ValueError("u 中包含 NaN 或 Inf")

    # 计算真正的右端项
    print("正在计算 rhs = B @ u ...")

    rhs = B @ u

    if rhs.shape != (nrow,):
        raise RuntimeError(f"rhs shape 错误：{rhs.shape}")

    if not np.isfinite(rhs).all():
        raise RuntimeError("rhs 中包含 NaN 或 Inf")

    np.save(RHS_OUTPUT, rhs)

    elapsed = time.time() - start_time

    print("=" * 70)
    print("转换与计算完成")
    print(f"B CSR 目录：{B_OUTPUT_DIR}")
    print(f"rhs 文件：{RHS_OUTPUT}")
    print(f"rhs shape：{rhs.shape}")
    print(f"rhs dtype：{rhs.dtype}")
    print(f"rhs 文件大小：{rhs.nbytes / 1024**3:.3f} GiB")
    print(f"rhs 非零数量：{np.count_nonzero(rhs):,}")
    print(f"rhs 最小值：{rhs.min():.16e}")
    print(f"rhs 最大值：{rhs.max():.16e}")
    print(f"rhs 前 10 项：{rhs[:10]}")
    print(f"总耗时：{elapsed:.1f} 秒")


if __name__ == "__main__":
    main()