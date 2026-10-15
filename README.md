# AIMoney

全自动量化交易系统 —— 研究、回测、自动交易一体化。当前完成 **P0 阶段（数据底座）**。

设计原则一句话：**engine 是纯计算内核，app 才是 IO 与编排**。这条约束决定了所有目录边界。

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

# 5) 起服务（可选）
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

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
```

- 策略目录约定：`strategy.py`（定义 `create_strategy(params) -> Strategy`）+ 可选 `config.yaml`；
- 输出：控制台摘要 + `data/backtests/<策略>_<时间戳>/` 下的 `metrics.json`、`nav.csv`、`trades.csv`；
- 回测严格模拟 A 股规则：T+1、涨跌停、停牌顺延、单笔 ≤ 当日成交量 10%、含费资金约束、固定滑点；
- 读 CSV 时记得 `dtype={"code": str}`——`"000001"` 会被推断成整数 `1`。

内置模板：`DualMovingAverageStrategy`（教学/冒烟）、`MomentumRotationStrategy`（定期调仓 + top_k 等权，接近实盘可用）。
多因子选股模板将在因子层进一步补齐后提供。

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

常用参数：`--root <目录>` 指定数据根目录，`--config <文件>` 指定配置文件，`--verbose` 打开 DEBUG 日志。

---

## 3. 接口

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
│   ├── cli.py            # 命令行入口（aimoney-data）
│   ├── main.py           # FastAPI 入口 + 异常到 HTTP 的映射
│   ├── api/v1/           # 路由层（只做参数校验与编排）
│   ├── services/         # 应用服务（业务语义、事务边界）
│   └── core/             # 配置 / 日志 / 加密 / 异常
├── engine/               # ★ 纯计算内核（禁止 import app）
│   ├── ports.py          # 端口契约（DataSource/BrokerAdapter/Context...）
│   ├── types.py          # 领域模型 + 订单状态机
│   ├── num.py  ids.py  errors.py
│   ├── datahub/          # 日历/列存/复权/股票池/质量/查询/同步/任务日志
│   ├── datasource/       # 数据源适配器（AkShare / 本地列存 / 内存替身）
│   ├── factor/           # 因子工厂（注册表/内置因子库/预处理/评价）
│   ├── strategy/         # 策略基类 + 加载器 + 内置模板
│   └── backtest/         # 回测内核（撮合/成本/指标/引擎主循环）
├── conf/                 # aimoney.yaml（可入库）/ secrets.yaml（已忽略）
├── data/                 # 运行时数据（已忽略）：parquet / reference / quality / state / logs
├── docs/                 # 需求 / 设计 / 任务 / 验收报告
├── strategies/           # 用户策略（P1 起）
└── tests/                # 399 条用例（另有联网用例默认跳过）
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
| `stock_info.list_date` 为空（akshare 免费接口不提供） | 「次新股过滤」会被标注为**未生效**（`UniverseSelection.rules_not_applied`），不会静默失效 | 接入 Tushare Pro / 券商行情源 |
| 涨跌停价无上游字段 | 由前收 × 板块阈值计算（T-09 已实现） | 如数据源提供则直接采用 |
| akshare 免费接口限流 | 短时间多次调用会 `RemoteDisconnected`，同步按标的隔离失败并计入告警 | 双源可切换（设计文档已有扩展点） |
| 分钟线 / 财务数据 | 未覆盖 | P3+ |
| 因子表达式引擎（自定义公式） | 只支持内置因子，暂不支持用户写表达式 | P2 后续 |
| 多因子合成与组合优化 | 已有相关性矩阵与评价，尚未做权重优化 | P2 后续 |

---

## 8. 免责声明

本项目为技术工具，不构成任何投资建议。历史回测表现不代表未来收益，实盘亏损风险由使用者自行承担。
禁止用于绕过券商官方接口、代客理财、承诺收益等违规用途。
