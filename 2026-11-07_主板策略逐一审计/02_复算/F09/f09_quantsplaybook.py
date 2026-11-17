# -*- coding: utf-8 -*-
"""F09 / QuantsPlaybook 参考库结论抽查（只读 GitHub API，不 clone）。
核验 2026-10-28 评估文档的关键事实：文件构成、无LICENSE、RSRS 前视、RRG shift(1)。
运行: python3 f09_quantsplaybook.py  （需要 gh CLI 已登录）
"""
import base64
import json
import os
import subprocess
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = "hugo2046/QuantsPlaybook"
rows = []


def gh(path):
    return subprocess.run(["gh", "api", path], capture_output=True, text=True,
                          check=True, timeout=120).stdout


def rec(item, claimed, actual, ok, note=""):
    rows.append(dict(项=item, 原称=claimed, 复核=actual, 判定=ok, 备注=note))
    print(f"[QP] {item}: 原称={claimed} 复核={actual} {ok} {note}")


meta = json.loads(gh(f"repos/{REPO}"))
rec("LICENSE", "无(null)", str(meta.get("license")), "一致" if meta.get("license") is None else "偏差")
rec("最近提交", "2026-10-03", meta["pushed_at"][:10],
    "一致" if meta["pushed_at"][:10] == "2026-10-03" else "偏差",
    f"star 现={meta['stargazers_count']}（审计时6,776，自然增长）")

tree = json.loads(gh(f"repos/{REPO}/git/trees/master?recursive=1"))["tree"]
blobs = [t for t in tree if t["type"] == "blob"]
nb = sum(t["path"].endswith(".ipynb") for t in blobs)
py = sum(t["path"].endswith(".py") for t in blobs)
pdf = sum(t["path"].endswith(".pdf") for t in blobs)
rec("文件构成", "2,183文件/65notebook/289py/83PDF",
    f"{len(blobs)}/{nb}/{py}/{pdf}",
    "一致" if (len(blobs), nb, py, pdf) == (2183, 65, 289, 83) else "偏差")


def blob_text(path):
    sha = next(t["sha"] for t in tree if t["path"] == path)
    b = json.loads(gh(f"repos/{REPO}/git/blobs/{sha}"))
    return base64.b64decode(b["content"]).decode("utf-8")


rsrs = json.loads(blob_text("C-择时类/RSRS择时指标/py/RSRS.ipynb"))
code = "\n".join("".join(c.get("source", [])) for c in rsrs["cells"] if c["cell_type"] == "code")
has_lookahead = ("Z_CUM = (1+RSRS_Z['z_singal']*RSRS_Z['ret']).cumprod()" in code
                 and "RSRS_Z['ret'] = RSRS_Z['close'].pct_change()" in code)
rec("RSRS T日信号吃T日收益(1日前视)", "存在", f"存在={has_lookahead}",
    "一致" if has_lookahead else "偏差",
    "同notebook的MA/BBANDS用shift(-1)正确口径；前视仅限RSRS系策略函数")

rrg = blob_text("C-择时类/相对旋转图RRG行业轮动/src/backtest.py")
ok_rrg = "weights_daily.shift(1)" in rrg and "fees: float = 3e-4" in rrg and "adjopen" in rrg
rec("RRG shift(1)+次日开盘+万3", "存在", f"存在={ok_rrg}",
    "一致" if ok_rrg else "偏差", "weights.shift(1)、price=adjopen(后复权开盘)、fees=3e-4")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "f09_quantsplaybook核查.csv"), index=False)
print(f"\n共 {len(rows)} 项，已写 f09_quantsplaybook核查.csv")