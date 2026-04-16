# cli.py 解耦重构计划

## Context

`cli.py` 有 10,044 行，单个 `HermesCLI` 类包含 ~120 个方法。`run()` 方法独占 1,854 行（含 40 个嵌套函数）。todo TUI 面板只是冰山一角——这个文件需要系统性地拆分。

现有代码已经有成功的抽取模式：`hermes_cli/callbacks.py`、`hermes_cli/commands.py`、`hermes_cli/banner.py` 都是从 cli.py 抽出来的独立模块，接收 `cli` 引用作为参数。

## 架构决策：组合 > Mixin

用组合模式（抽取独立模块，传 `cli` 引用），不用 Mixin。原因：
- Mixin 不解决 `self.` 耦合问题——每个 mixin 还能看到所有状态
- 组合强制显式接口，每个模块声明自己需要什么
- 现有代码已经用这个模式成功了

## HermesCLI 方法分组 & 行数

| 分组 | 行数 | 抽取优先级 |
|------|------|-----------|
| `run()` 主循环 | ~1854 | 最后拆（Phase 6-7） |
| Agent 交互/初始化 | ~1080 | Phase 5 |
| 斜杠命令处理 | ~870 | Phase 2 |
| 显示/渲染 (status bar, streaming) | ~915 | Phase 1 + 9 |
| 工具方法 (model picker, approval) | ~529 | Phase 8 |
| Voice 模式 | ~400 | Phase 3 |
| TUI 布局构建 | ~400 | Phase 6 |
| 配置/Profile | ~208 | Phase 4 |
| Session 管理 | ~150 | Phase 4 |
| Plugin/MCP | ~122 | 不单独拆 |

## 抽取顺序：从叶子节点开始

原则：先抽取依赖最少的模块，先抽纯函数，先写测试再移代码。

---

### Phase 0: 测试基础设施

**目标**：统一测试 fixture，后续所有 phase 共用。

- 创建 `tests/cli/conftest.py`
- `cli_stub()` fixture: `HermesCLI.__new__(HermesCLI)` + 最小属性
- `cli_with_agent()` fixture: 加 mock agent
- 提取 `_FakeAgent`、`_FakeBuffer` 到 `tests/cli/_helpers.py`

**产出**：`tests/cli/conftest.py` + `tests/cli/_helpers.py`

---

### Phase 1: 抽取 Streaming 显示 (~300 行)

**文件**：`hermes_cli/streaming.py`

**目标方法**（~2236-2640 行）：
- `_on_thinking`, `_current_reasoning_callback`
- `_emit_reasoning_preview`, `_flush_reasoning_preview`
- `_stream_delta`, `_emit_stream_text`, `_flush_stream`, `_reset_stream_state`

**为什么先抽这个**：耦合最低。只读 `self.show_reasoning`、`self.console.width`，自己的状态（`_stream_buf` 等）全内聚。

**TDD 步骤**：
1. 写 `tests/cli/test_stream_renderer.py` — 5 个测试覆盖缓冲、think tag 过滤、reasoning 合并
2. 创建 `hermes_cli/streaming.py` 的 `StreamRenderer` 类
3. `HermesCLI.__init__` 里 `self._stream = StreamRenderer(...)`
4. 替换调用点，删旧方法
5. 跑现有测试确认没破

**产出**：`hermes_cli/streaming.py` + 测试。cli.py 减 ~300 行。

---

### Phase 2: 抽取命令分发 (~870 行)

**文件**：`hermes_cli/command_dispatch.py`

**目标**：`process_command()` 的 370 行 if/elif 链 + 所有 `_handle_*` 方法

**TDD 步骤**：
1. 写 `tests/cli/test_command_dispatch.py` — 每个 `/command` 一个测试
2. 创建 `dispatch_command(cli, command: str) -> bool`
3. `process_command` 变成一行：`return dispatch_command(self, command)`
4. 逐步把 `_handle_*` 方法移到独立 handler 函数

**产出**：`hermes_cli/command_dispatch.py` + `hermes_cli/handlers/`。cli.py 减 ~870 行。

---

### Phase 3: 抽取 Voice 模式 (~400 行)

**文件**：`hermes_cli/voice_mode.py`

**目标方法**：`_voice_start_recording`, `_voice_stop_and_transcribe`, `_voice_speak_response`, `_handle_voice_command`, `_enable_voice_mode`, `_disable_voice_mode`, `_toggle_voice_tts`, `_show_voice_status`

**TDD 步骤**：
1. 写 `tests/cli/test_voice_controller.py`
2. 创建 `VoiceController` 类
3. `HermesCLI.voice = VoiceController(self)`
4. 替换调用点，删旧方法

**产出**：`hermes_cli/voice_mode.py` + 测试。cli.py 减 ~400 行。

---

### Phase 4: 抽取 Session 管理 (~350 行)

**文件**：`hermes_cli/session_manager.py`

**目标方法**：`new_session`, `_handle_resume_command`, `_handle_branch_command`, `save_conversation`, `retry_last`, `undo_last`, `_preload_resumed_session`, `_display_resumed_history`

**TDD 步骤**：
1. 写 `tests/cli/test_session_manager.py`
2. 创建 `SessionManager` 类
3. 替换调用点

**产出**：`hermes_cli/session_manager.py` + 测试。cli.py 减 ~350 行。

---

### Phase 5: 抽取 Agent 工厂 (~400 行)

**文件**：`hermes_cli/agent_factory.py`

**目标方法**：`_ensure_runtime_credentials`, `_normalize_model_for_provider`, `_resolve_turn_agent_config`, `_init_agent`

**注意**：耦合最高的抽取，需要读取/写入 provider、api_key、model 等核心状态。放第五是因为前四个先拆掉后，这部分会更清晰。

---

### Phase 6: 拆 run() — TUI 构建 (~600 行)

**文件**：`hermes_cli/tui_builder.py`

**目标**：从 `run()` 里抽出 keybinding 注册、layout 构建、style 配置。`run()` 从 1854 行降到 ~800 行。

---

### Phase 7: 拆 run() — 事件循环 (~400 行)

**文件**：`hermes_cli/process_loop.py`

**目标**：抽出 `process_loop()` 嵌套函数。`run()` 最终降到 ~200 行。

---

### Phase 8-10: 其余命令 handler 分组抽取

把 15+ 个 `_handle_*` 方法按功能分组到 `hermes_cli/handlers/` 下独立文件：

- `handlers/session.py` — rollback, snapshot, stop
- `handlers/model.py` — model switch, fast, reasoning
- `handlers/media.py` — image, paste, browser
- `handlers/tasks.py` — background, btw
- `handlers/config.py` — skin, personality, verbose, yolo
- `handlers/skills.py` — skills, cron

---

## 最终状态

cli.py 从 10,044 行降到 ~1,500-2,000 行，变成一个薄编排层：
- 持有配置状态
- 实例化各子模块
- 委托调用

新增 ~10 个模块文件 + ~15 个测试文件。

## 关键参考文件

- `hermes_cli/callbacks.py` — 已有的成功抽取模式
- `hermes_cli/commands.py` — `CommandDef` 注册表，Phase 2 会消费
- `tests/cli/test_cli_status_bar.py` — 最好的测试模式参考（`_make_cli()` + `_attach_agent()`）
- `tests/conftest.py` — `_isolate_hermes_home` fixture

## 验证方式

每个 Phase 独立可交付：
1. 写测试 → 测试通过
2. 抽取代码 → 所有测试（旧+新）通过
3. git commit，可随时停手

```bash
source venv/bin/activate
python -m pytest tests/cli/ -q          # 新测试
python -m pytest tests/ -q              # 全量回归
python -m pytest tests/test_cli_init.py -q  # CLI 特定
```
