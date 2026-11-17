"""事前冻结的有限策略族；不读行情，不据回测收益动态添加规则。"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Strategy:
    name: str
    trigger: str = "pullback"
    trigger_period: str = "d"
    structure_period: str = "w"
    frame_period: str = "m"
    structure_gate: bool = True
    frame_gate: bool = True
    line: str = "ema"
    quarter_gate: bool = False
    volume_gate: str = "none"
    market_gate: bool = False
    industry_gate: bool = False
    near_line: bool = False
    stop: str = "structure"
    exit: str = "support"
    slots: int = 4
    max_weight: float = 0.25
    risk_per_trade: float = 0.01
    heat_cap: float = 0.04
    max_stop_distance: float = 0.20
    rank: str = "momentum60"
    select: bool = True
    note: str = "研究者操作化，非原文唯一公式"


def registry():
    core = Strategy("月周日_回调_结构止损")
    return [
        replace(core, name="日突破_无上级过滤", trigger="breakout", structure_gate=False, frame_gate=False),
        replace(core, name="日回调_无上级过滤", structure_gate=False, frame_gate=False),
        replace(core, name="周日_回调", frame_gate=False),
        core,
        replace(core, name="月周日_突破", trigger="breakout"),
        replace(core, name="月周日_分型线", line="pivot"),
        replace(core, name="季月周日_回调", quarter_gate=True),
        replace(core, name="月周日_温和放量", volume_gate="moderate"),
        replace(core, name="月周日_缩量回调", volume_gate="contraction"),
        replace(core, name="月周日_市场过滤", market_gate=True),
        replace(core, name="月周日_市场行业过滤", market_gate=True, industry_gate=True),
        replace(core, name="季月周_回调", trigger_period="w", structure_period="m", frame_period="q"),
        replace(core, name="年季月_回调_描述性", trigger_period="m", structure_period="q", frame_period="y",
                select=False, note="年度框架预热长；不得用缺训练期的结果参与选择"),
        replace(core, name="月周日_近线", near_line=True),
        replace(core, name="二十日通道_基准", trigger="channel", structure_gate=False, frame_gate=False),
        replace(core, name="月周日_小周期止损", stop="trigger"),
        replace(core, name="月周日_固定百分之八", stop="fixed8"),
        replace(core, name="月周日_两倍波幅", stop="atr2"),
        replace(core, name="月周日_均线退出", exit="line"),
        replace(core, name="月周日_二十日退出", exit="time20"),
        replace(core, name="月周日_超买减半", exit="partial"),
        replace(core, name="月周日_背离退出", exit="divergence"),
        replace(core, name="月周日_两仓", slots=2, max_weight=0.50),
        replace(core, name="月周日_六仓", slots=6, max_weight=1 / 6),
        replace(core, name="月周日_等仓_结构止损", risk_per_trade=1.0, heat_cap=1.0,
                select=False, note="诊断止损效果与风险仓位混淆，不参与择优"),
        replace(core, name="月周日_等仓_小周期止损", stop="trigger", risk_per_trade=1.0, heat_cap=1.0,
                select=False, note="固定仓位止损配对诊断"),
        replace(core, name="月周日_等仓_固定百分之八", stop="fixed8", risk_per_trade=1.0, heat_cap=1.0,
                select=False, note="固定仓位止损配对诊断"),
        replace(core, name="月周日_等仓_两倍波幅", stop="atr2", risk_per_trade=1.0, heat_cap=1.0,
                select=False, note="固定仓位止损配对诊断"),
    ]


if __name__ == "__main__":
    path = ROOT / "04_证据" / "策略注册表.json"
    if path.exists():
        current = json.loads(path.read_text())
        if current != [asdict(x) for x in registry()]:
            raise RuntimeError("注册表已存在且不一致；需登记方案变更，不允许静默覆盖")
    else:
        path.write_text(json.dumps([asdict(x) for x in registry()], ensure_ascii=False, indent=2))
    print(f"策略共{len(registry())}组；参与预先选择{sum(x.select for x in registry())}组；{path}")
