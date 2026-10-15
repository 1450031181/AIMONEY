# AIMoney

全自动量化交易系统 —— 研究、回测、自动交易一体化。设计原则一句话：
**engine 是纯计算内核，app 才是 IO 与编排**。这条约束决定了所有目录边界。

## 0. 当前状态（2026-10-15）

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 数据底座 | 同步 / 复权 / 股票池 / 质量门禁 / DuckDB 查询 / CLI / HTTP | ✅（真实历史全量回填受数据源限流阻塞） |
| P1 回测闭环 | 撮合（T+1/涨跌停/停牌/量约束）、成本滑点、绩效指标、策略模板、`aimoney-backtest` | ✅ |
| P2 因子工厂 | 21 个内置因子、预处理、IC/ICIR/分层/换手评价、多因子合成、因子快照接入策略、`aimoney-factor` | ✅ |
| P3 组合层 | 分配方案（等权/得分/排名/逆波动率）+ 上限/碎仓/无交易带/换手约束 | ✅ |
| P4 风控 | 事前 8 条 + 事中 3 条规则、急停开关、回测与实盘同一条链 | ✅ |
| P5 实盘链路 | 模拟盘券商、订单管理（幂等/闸门/查单校准）、账本持久化、`aimoney-paper`、`aimoney-schedule` | ✅（模拟盘，非真实券商） |
| P6 进阶 | 组合优化、参数寻优、Walk-Forward、AI 因子 | ⬜ |

未实现的部分集中在「前端 / 告警 / 真实券商 / 盘中高频 / 分钟线与财务数据」——
完整清单与原因见 [§7 已知缺口](#7-已知缺口)。**593 条用例**，`ruff` 与 `mypy --strict`（82 个源文件）全绿。

五个命令行入口：`aimoney-data`（数据）、`aimoney-backtest`（回测）、`aimoney-factor`（因子）、
`aimoney-paper`（模拟盘）、`aimoney-schedule`（盘前/盘后调度）。

---

## 1. 快速开始

```powershell
# 1) 环境（Python 3.11+；本项目在 3.13 上开发验证）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -U pip
.venv\Scripts\python.exe -m pip install -e ".[data,source,web,dev]"

# 2) 敏感配置（可选，一期用模拟盘/公开数据源时不需要）
copy conf\secrets.example.yaml conf\secrets.yaml
.venv\Scripts\python.exe -m app.core.crypto keygen   # 生成主密钥
$env:AIMONEY_MASTER_KEY = "<上一步输出>"

# 3) 数据初始化：同步 → 质量校验 → 查看覆盖
.venv\Scripts\aimoney-data.exe sync --start 2026-09-01 --end 2026-10-15 --full --codes 600000,000001
.venv\Scripts\aimoney-data.exe check --date 2026-10-15
.venv\Scripts\aimoney-data.exe coverage

# 4) 跑第一个回测（内置双均线示例策略，报告落盘 data/backtests/）
.venv\Scripts\aimoney-backtest.exe --strategy strategies/dual_moving_average `
    --start 2026-09-01 --end 2026-10-22 --codes 600000 --params fast=3 --params slow=10

# 5) 多因子：先评价因子，再跑多因子策略回测（必须用 --factors 把因子喂给回测）
.venv\Scripts\aimoney-factor.exe evaluate --factors mom_20,rev_5,vol_20 `
    --start 2026-09-01 --end 2026-10-12
.venv\Scripts\aimoney-backtest.exe --strategy strategies/multi_factor_enhance `
    --start 2026-08-17 --end 2026-10-09 --factors mom_20,vol_20

# 6) 模拟盘：盘前（同步 + 质量校验）与盘后（跑一天 + 查单校准），定时任务每天各调一次
.venv\Scripts\aimoney-schedule.exe once --job pre  --date 2026-10-15 --codes "600000,000001"
.venv\Scripts\aimoney-schedule.exe once --job post --date 2026-10-15 `
    --strategy strategies/multi_factor_enhance --factors mom_20,vol_20

# 7) 起服务（可选）
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> 注意 PowerShell 会**吃掉前导零**：`--codes 600000,000001` 会被解析成 `[600000, 1]`。
> 给列表加引号，或重复传参（`--codes 000001`）。

Linux/macOS 把 `.venv\Scripts\` 换成 `.venv/bin/`。

---

## 2. 命令行（`aimoney-data`）

| 子命令 | 作用 | 典型用法 |
| --- | --- | --- |
| `list` | 列出本地所有数据集及覆盖情况 | `aimoney-data list` |
| `coverage` | 单个数据集详细覆盖 + 交易门禁状态 | `aimoney-data coverage --json` |
| `sync` | 日线增量同步（含真实复权因子） | `aimoney-data sync --start 2018-01-01 --full` |
| `index-members` | 抓取指数当前成分并按抓取日打点 | `aimoney-data index-members --index 000300,000905` |
| `check` | 数据质量校验，写日报并开关交易门禁 | `aimoney-data check --date 2026-10-15 --json` |

**退出码是调度契约**：`0` 正常 / `1` 数据侧问题（同步有失败标的、质量 FAIL）/ `2` 用法错误。
定时任务与 CI 依赖它判断"今天的数据能不能用"。

## 2.1 回测（`aimoney-backtest`）

```powershell
aimoney-backtest --strategy strategies/dual_moving_average `
    --start 2026-09-01 --end 2026-10-22 `
    [--codes 600000] [--cash 1000000] [--slippage-bps 5] `
    [--params fast=3 --params slow=10] [--json]

# 多因子策略：必须用 --factors 把因子喂给回测（否则策略会明确报错）
aimoney-backtest --strategy strategies/multi_factor_enhance `
    --start 2026-08-17 --end 2026-10-09 --factors mom_20,vol_20
```

- 策略目录约定：`strategy.py`（定义 `create_strategy(params) -> Strategy`）+ 可选 `config.yaml`；
- 输出：控制台摘要 + `data/backtests/<策略>_<时间戳>/` 下的 `metrics.json`、`nav.csv`、`trades.csv`；
- 回测严格模拟 A 股规则：T+1、涨跌停、停牌顺延、单笔 ≤ 当日成交量 10%、含费资金约束、固定滑点；
- 读 CSV 时记得 `dtype={"code": str}`——`"000001"` 会被推断成整数 `1`。

内置模板：`DualMovingAverageStrategy`（教学/冒烟）、`MomentumRotationStrategy`（动量轮动）、
`MultiFactorEnhanceStrategy`（多因子选股，需 `--factors` 提供因子）。

**因子如何进入策略**：回测引擎把预计算的因子值按**当日切片**注入 `Context`，
策略通过 `context.factor(name, code)` 取用——它拿不到其他日期的因子，
越界在接口上不存在，而不是靠约定。缺值返回 `None`（预热期、停牌、数据缺口都属正常，
策略可显式处理）；只有"回测压根没装载因子"才会明确报错。

## 2.2 因子工厂（`aimoney-factor`）

```powershell
# 查看内置因子库（21 个量价因子，六类：动量/反转/波动/趋势/流动性/结构）
aimoney-factor list [--category momentum] [--json]

# 计算并评价因子：IC / ICIR / 分层回测 / 多空组合 / 换手 / 单调性 / 因子相关性
aimoney-factor evaluate --factors mom_20,rev_5,vol_20 `
    --start 2026-09-01 --end 2026-10-12 [--horizon 1] [--quantiles 5] [--json]
```

- 报告落盘到 `data/factors/evaluate_<时间戳>/`：`summary.json`、`ic_<因子>.csv`、
  `quantile_<因子>.csv`（各层净值）、`correlation.csv`；
- **自动处理预热期**：`mom_60` 需要 61 根 K 线，工具会自动多读 61 个交易日的数据再评价，
  否则评价窗口开头会有一段因子值为 NaN、被静默跳过；
- 默认预处理 `winsorize,rank,zscore`（`--preprocess none` 可关闭）。

判据怎么读：**IC** 是预测力，**ICIR** 是稳定性，**单调性** 说明因子在真的给标的排序而不是靠某一层运气，
**换手率** 决定 IC 有多少会被交易成本吃掉——四个要一起看。

## 2.3 多因子合成（`aimoney-factor combine`）

```powershell
aimoney-factor combine --factors mom_20,rev_5,vol_20 `
    --start 2026-08-03 --end 2026-10-09 `
    [--method icir|ic|equal] [--weight-window 20] [--top-k 10] [--correlation-threshold 0.7] [--json]
```

产出：合成因子的评价 + 与最强单因子的对比 + 每日权重 + 选股名单，落盘
`data/factors/combine_<时间戳>/`（`summary.json`、`weights.csv`、`selections.csv`、`ic_*.csv`）。

三条关键规则：

1. **权重只用当时可见的信息**：默认按滚动窗口的 ICIR 估计，`t` 日权重仅由
   `[t-窗口, t-1]` 的 IC 决定，窗口内的截面权重为空（不可用）——
   用全样本 IC 定权重是量化里最典型的未来函数，会让合成因子的 IC 被系统性高估。
2. **先去重再估权重**：相关性超过阈值的因子只保留 |IC| 更强的那个，
   否则等于把同一个信号按满仓位重复押注。
3. **IC 为负的因子会被反向使用**（权重为负），而不是丢弃。

> 实测提醒：合成**不一定**跑赢最强单因子。演示数据上 `mom_20` 单因子 IC 0.22，
> 而 ICIR 合成后为 0.14——因为滚动权重有估计滞后，且被反向使用的 `vol_20` 权重过大。
> 工具会如实报出这种负面结论（摘要里直接给"IC 变化"），而不是挑好看的指标展示。

## 2.4 组合约束（策略配置项）

选股名单到真实下单之间还有一步：**目标权重**。在策略的 `config.yaml` 里配置（默认全部关闭 = 等权满仓）：

| 参数 | 作用 | 不配的后果 |
| --- | --- | --- |
| `weight_scheme` | `equal` / `score`（按得分加权）/ `rank` / `inverse_vol` | 强弱候选一视同仁 |
| `max_weight` | 单标的上限 | 一只票的坏消息吃掉整个组合 |
| `min_weight` | 剔除碎仓 | 最低 5 元手续费比碎仓的贡献还高 |
| `no_trade_band` | 权重变化小于阈值就不动 | 每天为 0.1% 的权重差付一次手续费 |
| `max_turnover` | 单次调仓换手上限 | 一次信号抖动就把组合翻个底朝天 |

两个容易踩的坑，都已按正确语义实现并被测试守住：**上限截断与碎仓剔除后必须把权重重新分配**，
否则总仓位会悄悄少一截、凭空多出现金敞口且不报任何错；**容量确实不足时（如 3 只 × 上限 20%）
就老实留现金**，而不是把上限顶破。

## 2.5 风控（`conf/aimoney.yaml` 的 `risk` 段）

回测与实盘共用同一套规则，**默认开启**：策略拿不到下单能力（`Context` 里没有下单方法），
所有计划都必须过风控闸门。

| 配置 | 作用 | 触发效果 |
| --- | --- | --- |
| `max_weight` | 单票集中度上限 | **缩量**到上限（不是拒单：信号可能没问题，只是仓位太大） |
| `max_order_value` | 单笔委托金额上限 | 缩量 |
| `max_orders_per_day` | 当日委托笔数上限 | 拒单 |
| `max_drawdown` | 组合回撤触线 | `HALT`：只平不开 |
| `max_daily_loss` | 单日亏损触线 | `HALT` |
| `liquidation_drawdown` | 深度回撤 | `FLATTEN`：清仓 |

三条设计要点：

1. **降级必须同时撤回未成交的买单**。只拦"新计划"是不够的——昨天挂出的买单今天照样成交，
   等于降级慢一拍。回测里事中检查因此排在撮合**之前**（早上已知昨日收盘的回撤，今天就不该再买）。
2. **卖出尽量放行**：风控的目标是控制风险暴露，降低暴露的操作不该被拦。
3. **两个安全机制的默认值刻意相反**：交易门禁 `trade_gate` 文件缺失时**关闭**（没跑过数据质量校验就不许开车），
   急停开关 `kill_switch` 文件缺失时**未激活**（紧急制动当然不该默认踩着）；
   但两者读取异常时都倒向安全侧——急停文件损坏即视为已激活。

> 回测报告会如实记录 `risk.rejections` / `risk.shrinks` / `risk.level_events`，
> 并在摘要里显示"缩量 N 笔、拒单 M 笔、触线降级 K 次"——风控拦下的东西必须看得见，
> 否则"信号被吞掉"会变成无人知晓的沉默成本。

**默认值要按持仓只数调整**：`max_weight` 默认 20%，若策略只持 3 只（每只约 32%），
上限会持续截断、实际仓位远低于预期（实测：默认配置下缩量 14 笔、总收益 7.64%，
关闭风控则 12.28%）。这不是 bug 而是配置与组合规模不匹配——
**持仓 N 只时 `max_weight` 至少要 ≥ 1/N**，否则等于强制留大量现金。

## 2.6 实盘链路（模拟盘先行）

`engine/broker/paper.py` + `engine/trade/oms.py` 让**实盘链路的全部代码在离线状态下被完整执行**：
风控 → 报单 → 冻结 → 撮合 → 成交通知 → 持仓资金更新 → 日切 → 查单校准。
若这些路径只在实盘第一次运行，第一次运行就是"带着真钱调试"。

**交易日循环的时序**（顺序错了会出真实问题）：

```
1. 推送当日行情 → 券商更新参考价，并撮合昨日挂单
2. 信号 → 组合权重 → 事前风控 → 报单（按当日价冻结资金）
3. 日切（T+1 解冻）→ 记录净值
```

第 1 步必须在第 2 步之前：冻结资金需要参考价，没有参考价的市价单会被明确拒绝
（拿臆测价格冻结资金会导致超额委托）。

三条与真实券商对齐的语义，都有测试钉住：

| 语义 | 说明 |
| --- | --- |
| **报单幂等** | 同一 `Order.idem_key` 只产生一笔委托。幂等键由 `策略\|日期\|代码\|方向\|数量` **确定性**推导——用随机的计划号在重启后就失效，同一笔单会被重复报出 |
| **冻结守恒** | 撤单/废单/顺延超限都必须归还冻结的资金与持仓，否则会缓慢泄漏（实测抓出"先记账后冻结失败"导致的解冻越界） |
| **异常不重报** | 通道异常标记 `UNKNOWN` 并走查单校准，绝不自动重报 |

> 模拟盘券商的撮合复用回测的同一套规则（T+1/涨跌停/停牌/量约束/资金约束），
> 因此"回测能成交的形态"与"模拟盘能成交的形态"一致，不会出现两套口径。

## 2.7 模拟盘运行（`aimoney-paper`）

```powershell
# 每天调用一次即可（建议配合定时任务）
aimoney-paper run --strategy strategies/multi_factor_enhance --date 2026-10-12 [--json]
aimoney-paper status --strategy strategies/multi_factor_enhance
```

**一个交易日的完整时序**（顺序即正确性）：

```
0. 日切（T+1 在日初解冻）→ 1. 推送行情并撮合昨日挂单 → 2. 策略决策
→ 3. 风控 → 报单（对下一交易日有效）→ 4. 盯市与落盘
```

| 退出码 | 含义 |
| --- | --- |
| `0` | 正常 |
| `1` | 出错 |
| `3` | **跳过**（门禁未开 / 急停 / 当日无行情 / 该交易日已处理） |

把"跳过"与"出错"分开是刻意的：门禁未开是预期内的状态，共用一个退出码会让监控很快对告警脱敏。

三条运维事实：

1. **日级防重**：重复跑同一天**不是幂等的**——上次的挂单会在同一天行情上再撮一次，
   产生本不该有的成交。运行器因此记录"最后处理日"，重复即跳过；`--force` 可覆盖，
   而那正是它不该进定时任务的原因。
2. **账本要存全**：账户、持仓、**在途委托的冻结台账**、委托流水都必须落盘。
   少存任何一项都会出真实问题——只存账户会让昨天的挂单今天无人撮合，
   冻结资金变成永远解不开的悬空额度（实测过）。
3. **能力边界**：JSON 账本无事务、无并发控制、无对账凭证，只适合单人单进程模拟盘；
   接真实资金前必须换成事务型存储。这条写在 `engine/trade/ledger.py` 的模块文档里，
   避免它被误当成生产级账本。

## 2.8 定时调度（`aimoney-schedule`）

```powershell
# 盘前（交易日早间）：同步最新行情 → 质量校验 → 开关门禁
aimoney-schedule once --job pre  --date 2026-10-15 --codes "600000,000001"
# 盘后（交易日收盘后）：跑模拟盘一天 → 查单校准
aimoney-schedule once --job post --date 2026-10-15 `
    --strategy strategies/multi_factor_enhance --factors mom_20,vol_20
# 核对接下来会触发的交易日（验证日历守卫没失效）
aimoney-schedule plan --days 7
```

**不引入常驻进程**：交给 Windows 任务计划 / cron 每天调一次即可，退出码同前（`0` 正常 / `1` 有步骤失败 / `3` 跳过）。

四条必须知道的行为：

1. **交易日守卫以数据源为准，而不是本地行情**。本地日历等价于"行情数据里出现过的日期"，
   而**盘前那一刻今天的数据本来就不存在**——用本地日历会把今天判成非交易日，
   于是盘前任务永不触发、数据永远同步不了（死锁，实测踩到过）。
   数据源不可用（网络/限流）时才降级为本地推导并打 WARNING。
2. **同步失败不阻断质量校验**：否则门禁会停在旧状态，出现"数据坏了却还开着闸门"这种最危险的组合。
   失败仍让退出码非零，只是不妨碍门禁更新。
3. **幂等空转不算失败**：本地已是最新时同步会跳过，这是成功——当失败处理的话，
   定时任务从第二天起每天都会误告警，监控很快就被脱敏。
4. **质量 FAIL 不算调度失败**：检查本身成功执行了，结论是"今天不该交易"，门禁会拦住交易；
   若这里也报错，真正的执行故障会被刷屏噪声淹没。

> **PowerShell 会吃掉前导零**（本机实测踩到）：`--codes 600000,000001` 会被解析成 `[600000, 1]`。
> 质量校验随后判定"缺失 3/8 只"并关闭门禁——**判决是对的，但原因很隐蔽**。
> 给列表加引号，或重复传参（`--codes 000001`）。

常用参数：`--root <目录>` 指定数据根目录，`--config <文件>` 指定配置文件，`--verbose` 打开 DEBUG 日志。

---

## 3. 接口

当前**只有数据相关接口**（因子/回测/交易/风控/报告等 REST 路由尚未实现，对应能力目前只经 CLI 提供）：

| 方法与路径 | 说明 |
| --- | --- |
| `GET /health` | 健康检查（含门禁状态） |
| `GET /api/v1/data/coverage?dataset=daily` | 数据覆盖情况与门禁 |
| `GET /api/v1/data/gate` | 交易门禁状态 |
| `GET /api/v1/data/quality/{as_of}` | 指定日期的质量报告（未跑过则 404） |

启动后可访问 `http://127.0.0.1:8000/docs` 查看自动生成的接口文档。

---

## 4. 目录结构

```
AIMoney/
├── app/                  # 应用层：IO、装配、编排
│   ├── cli.py            # aimoney-data（list/coverage/sync/index-members/check）
│   ├── backtest_cli.py   # aimoney-backtest
│   ├── factor_cli.py     # aimoney-factor（list/evaluate/combine）
│   ├── paper_cli.py      # aimoney-paper（run/status）
│   ├── schedule_cli.py   # aimoney-schedule（once/plan）
│   ├── main.py           # FastAPI 入口 + 异常到 HTTP 的映射
│   ├── api/v1/           # 路由层（只做参数校验与编排；当前仅 data）
│   ├── services/         # 应用服务：data / backtest / factor / paper / schedule
│   └── core/             # 配置 / 日志 / 加密 / 异常
├── engine/               # ★ 纯计算内核（禁止 import app）
│   ├── ports.py          # 端口契约（DataSource/BrokerAdapter/Context...）
│   ├── types.py          # 领域模型 + 订单状态机
│   ├── num.py  ids.py  errors.py
│   ├── datahub/          # 日历/列存/复权/股票池/质量/查询/同步/任务日志
│   ├── datasource/       # 数据源适配器（AkShare / 本地列存 / 内存替身）
│   ├── factor/           # 因子工厂（注册表/内置因子库/预处理/评价/合成/快照）
│   ├── portfolio/        # 组合层（分配方案/约束/组合构建）
│   ├── strategy/         # 策略基类 + 加载器 + 内置模板
│   ├── risk/             # 风控（事前逐单 / 事中触线 / 急停开关）
│   ├── broker/           # 券商适配器（模拟盘：冻结/撮合/回报）
│   ├── trade/            # 交易链路（订单管理：幂等报单/闸门/查单校准）
│   └── backtest/         # 回测内核（撮合/成本/指标/引擎主循环）
├── conf/                 # aimoney.yaml（可入库）/ secrets.yaml（已忽略）
├── data/                 # 运行时数据（已忽略）：parquet / reference / quality / state / logs
├── docs/                 # 需求 / 设计 / 任务 / 验收报告
├── strategies/           # 用户策略（P1 起）
└── tests/                # 593 条用例（另有联网用例默认跳过）
```

---

## 5. 开发

```powershell
.venv\Scripts\python.exe -m pytest            # 全量用例（默认排除联网）
.venv\Scripts\python.exe -m pytest -m network # 联网冒烟（数据源接口）
.venv\Scripts\python.exe -m ruff check .      # 静态检查
.venv\Scripts\python.exe -m mypy              # 类型检查（strict）
```

约定：

- 价格/金额一律 `Decimal`，禁止 float 参与资金累加；批量行情用 pandas（与数据源对齐）；
- 新增数据源/券商：实现对应 `Protocol` 即可，引擎零改动；
- 策略拿到的 `Context` **没有下单方法**，只能产出 `Signal`（由测试守护）；
- 提交前 `ruff` 与 `mypy` 必须干净。

---

## 6. 四条硬规则（已写进代码并由测试守住）

1. **复权三口径分离**：回测计算用后复权、下单用真实价（`后复权 ÷ F(t)`，按 0.01 取整）、展示用前复权。因子缺失/非正一律报错，**绝不按 1 兜底**。
2. **防未来函数**：数据读取受"当前时点"约束、财务按公告日对齐、指数成分按生效日解析（没有快照的日期直接报错，不用最新成分回填）。
3. **交易门禁 fail-safe**：`data/state/trade_gate.json` 不存在时门禁**默认关闭**；数据质量 FAIL 立即关门禁——宁可当日空仓，不拿脏数据下单。
4. **幂等与原子**：同步按 `(code, date)` 幂等 upsert，落盘先写 `.tmp` 再替换；下单幂等键 `idem_key` 保证重试不重复报单。

---

## 7. 已知缺口

| 缺口 | 影响 | 计划 |
| --- | --- | --- |
| **真实券商适配器（QMT / miniQMT）** | 只能跑模拟盘；`BrokerAdapter` 端口与幂等/校准机制已就绪，缺的只是适配器与权限 | 需券商权限与环境（P5） |
| **Web 前端与图表页** | 报告只能看 `metrics.json` / CSV；`frontend/` 目前仅占位 | 未排期 |
| **告警通道（钉钉 / 企业微信）** | 任务失败只能靠结构化日志、任务日志与退出码发现 | 未排期 |
| **盘中高频任务** | 当前只在日频跑一次（盘前同步 + 盘后下单），未做盘中信号与事中风控采样 | 依赖盘中行情 |
| `stock_info.list_date` 为空（akshare 免费接口不提供） | 「次新股过滤」会被标注为**未生效**（`UniverseSelection.rules_not_applied`），不会静默失效 | 接入 Tushare Pro / 券商行情源 |
| 涨跌停价无上游字段 | 由前收 × 板块阈值计算（T-09 已实现） | 如数据源提供则直接采用 |
| akshare 免费接口限流 | 短时间多次调用会 `RemoteDisconnected`，同步按标的隔离失败并计入告警 | 双源可切换（设计文档已有扩展点） |
| 分钟线 / 财务数据 | 未覆盖（T-14 / T-15 加档项） | P6 |
| 因子表达式引擎（自定义公式） | 只支持内置 21 个量价因子，暂不支持用户写表达式 | P2 后续 |
| 组合权重优化（均值方差 / 风险平价） | 组合层只做分配方案与约束，未做最优化求解 | P6 |
| 收益归因（`attribution.py`） | 只有绩效指标，没有"收益来自哪些标的/因子"的分解 | 未排期 |
| APScheduler 常驻调度 | 需要外部调度器（任务计划 / cron）每天调用 | 按需 |

---

## 8. 免责声明

本项目为技术工具，不构成任何投资建议。历史回测表现不代表未来收益，实盘亏损风险由使用者自行承担。
禁止用于绕过券商官方接口、代客理财、承诺收益等违规用途。
