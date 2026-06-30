# 本地模型嵌入作为智能体 — 综合方案 v2

> 纯本地、零服务器、零外部API的浏览器扩展方案
> 
> 审查轮次：v1 → 审查员 #1（架构确认）→ v2 整合所有反馈

---

## 一、需求概述

在配置文件中修改本地模型的名称及路径，模型自动映射到固定页面上，每个页面有强约束配置文件决定模型角色和准则，实现本地模型自由嵌入网页作为智能体。

### 硬约束

- ❌ 无服务器 — 不能部署后端
- ❌ 外部 API 不可用 — 中国大陆网络环境（OpenAI/Claude 等被墙）
- ✅ Windows 平台
- ✅ 完全本地运行

---

## 二、架构设计

```
┌─────────────────────────────────────────────────┐
│  浏览器页面 (GitHub / 知乎 / 任意网页)              │
│  ┌───────────────────────────────────────────┐  │
│  │  浮动智能体面板 (content.js)                 │  │
│  │  - Shadow DOM 隔离 (mode: closed)          │  │
│  │  - 流式 token 逐字渲染 (打字机效果)          │  │
│  │  - Ollama 离线状态提示                     │  │
│  └──────────────┬────────────────────────────┘  │
│                 │ chrome.runtime.sendMessage      │
│                 │ (短问答) / connect port (流式)   │
│  ┌──────────────▼────────────────────────────┐  │
│  │  Extension Service Worker (background.js)  │  │
│  │  - URL 模式匹配 → 角色映射                   │  │
│  │  - 请求队列 (多标签页并发控制)                │  │
│  │  - 连接状态检查 (Ollama 存活探测)            │  │
│  │  - 输出三层校验 (见第四节)                   │  │
│  └──────────────┬────────────────────────────┘  │
└─────────────────┼────────────────────────────────┘
                  │ HTTP localhost:11434
┌─────────────────▼────────────────────────────────┐
│  Ollama (本地大模型推理服务)                       │
│  - qwen2.5:7b / deepseek-r1:8b / glm4:9b 等     │
│  - format: json (约束第三层 — 注意其局限性)        │
│  - stream: true (流式输出)                        │
│  - 127.0.0.1:11434 (仅本机可达，无鉴权)           │
└──────────────────────────────────────────────────┘
```

### 设计决策

| 决策 | 理由 |
|------|------|
| 去 LangChain | 无服务器跑不了 Python；JS 版 500KB+ 过度抽象，Ollama HTTP API 足够 |
| 浏览器扩展而非独立应用 | 天然嵌入网页，无需额外安装运行时 |
| Ollama 而非 llama.cpp 直接调用 | Ollama 提供标准 HTTP API + 模型管理，开发体验好得多 |
| JSON 而非 YAML 配置 | 浏览器无原生 YAML 解析，JSON 零依赖 |
| 国产中文模型为主力 | qwen2.5 / deepseek-r1 中文能力强，Ollama 可直接 pull |

---

## 三、配置文件设计

> **v2 修订**：YAML → JSON（零依赖），新增 `num_ctx`、`roles.json`（角色复用），添加 few-shot examples。

### `models.json` — 模型注册

```json
{
  "models": [
    {
      "name": "code-helper",
      "ollama_model": "qwen2.5:7b",
      "temperature": 0.3,
      "max_tokens": 4096,
      "num_ctx": 8192
    },
    {
      "name": "doc-reader",
      "ollama_model": "deepseek-r1:8b",
      "temperature": 0.1,
      "max_tokens": 2048,
      "num_ctx": 4096
    },
    {
      "name": "general-chat",
      "ollama_model": "qwen2.5:7b",
      "temperature": 0.7,
      "max_tokens": 2048,
      "num_ctx": 4096
    }
  ]
}
```

**关键字段说明**：
- `num_ctx`：上下文窗口大小。Ollama 默认仅 2048 token，代码审查等长上下文场景必须加大。
- `max_tokens`：单次回复最大长度。
- `temperature`：低值（0.1-0.3）适合代码/文档场景，高值（0.7+）适合聊天。

### `roles.json` — 可复用的角色模板

> **v2 新增**：避免在每个 page 条目中重复 `output_filter` 和 `response_schema`，多页面可引用同一角色。

```json
{
  "roles": {
    "code-reviewer": {
      "name": "代码助手",
      "system_prompt": "你是代码审查助手。严格规则：\n- 只讨论代码，拒绝非代码话题\n- 分析代码逻辑并给出具体改进建议\n- 永远不要建议执行命令或修改文件\n- 用中文回答\n\n输出格式示例（必须严格遵守）：\n{\"type\": \"code_review\", \"content\": \"这里可以优化...\", \"line_references\": [42, 58]}",
      "output_filter": {
        "forbidden_patterns": [
          "执行.*命令",
          "rm\\s+-rf",
          "drop\\s+table",
          "delete\\s+from",
          "sudo\\s+"
        ]
      },
      "response_schema": {
        "type": "object",
        "required": ["type", "content"],
        "properties": {
          "type": { "enum": ["code_review", "explanation", "refusal"] },
          "content": { "type": "string" },
          "line_references": { "type": "array", "items": { "type": "integer" } }
        }
      }
    },
    "doc-reader": {
      "name": "阅读助手",
      "system_prompt": "你是阅读助手。严格规则：\n- 总结和解释当前页面的内容\n- 不做主观评价\n- 问题超出页面范围时，明确告知用户\n\n输出格式示例（必须严格遵守）：\n{\"type\": \"summary\", \"content\": \"本文主要讨论了...\", \"in_scope\": true}",
      "output_filter": {
        "forbidden_patterns": [
          "我认为",
          "我觉得",
          "你应该"
        ]
      },
      "response_schema": {
        "type": "object",
        "required": ["type", "content", "in_scope"],
        "properties": {
          "type": { "enum": ["summary", "explanation", "refusal"] },
          "content": { "type": "string" },
          "in_scope": { "type": "boolean" }
        }
      }
    }
  }
}
```

**关于 few-shot examples**：审查员指出，7b-8b 级别的小模型 JSON Schema 遵循率只有 75-85%。在 system prompt 中提供**具体格式示例**比只写 schema 定义更有效。上方的 `system_prompt` 已内含格式示例。

### `pages.json` — 页面 → 角色绑定

```json
{
  "pages": [
    {
      "url_pattern": "*://github.com/*",
      "model_ref": "code-helper",
      "role_ref": "code-reviewer"
    },
    {
      "url_pattern": "*://*.zhihu.com/*",
      "model_ref": "doc-reader",
      "role_ref": "doc-reader"
    }
  ]
}
```

**职责划分**：
```
models.json  → "运行什么"（模型名、temperature、num_ctx、max_tokens）
roles.json   → "扮演谁、输出什么格式"（system prompt、output_filter、schema）
pages.json   → "什么时候触发"（URL pattern → model_ref + role_ref）
```

---

## 四、三层安全约束体系（v2 修订）

> **v2 重要修订**：审查员指出 Ollama 的 `format: json` 只保证输出是合法 JSON，**不保证符合你的 schema**。第二层需要承担远比"正则拦截"更重的职责。

### JSON Schema 现实检查

| 模型 | JSON 格式正确率 | Schema 完全符合率 |
|------|:--:|:--:|
| qwen2.5:7b | ~95% | ~80-85% |
| deepseek-r1:8b | ~90% | ~75-80% |
| qwen2.5:14b | ~98% | ~90% |

DeepSeek-R1 特别需要注意 — 其思维链（`<think>` 标签）有时会泄漏到 JSON 输出中破坏结构。

### 三层防护

```
第一层 ─ System Prompt + Few-Shot Example（软约束）
  ├─ 模型收到角色指令，尽力遵守
  └─ 内嵌 JSON 格式示例，帮助小模型理解输出格式

第二层 ─ output-filter.js（中约束）★ v2 职责加重
  ├─ 步骤 1: JSON.parse() — 捕获模型输出了非法 JSON 的情况
  ├─ 步骤 2: Schema 校验（required 字段、enum 值、类型检查）
  ├─ 步骤 3: 正则黑名单匹配 forbidden_patterns
  └─ 任一失败 → 返回错误，触发重试或降级

第三层 ─ Ollama format: json（硬约束）
  ├─ 底层 llama.cpp grammar 约束，模型只能输出 JSON 语法合法的内容
  └─ 局限性：不验证 JSON 结构、不保证字段完整
```

### output-filter.js 核心实现

```javascript
// lib/output-filter.js
class OutputFilter {
  /**
   * @param {object} roleCfg - 来自 roles.json 的角色配置
   * @param {string} raw - Ollama 返回的原始文本
   * @returns {{ valid: boolean, data?: object, error?: string, raw?: string }}
   */
  static validate(roleCfg, raw) {
    // 步骤 1: JSON 解析 — 捕获坏 JSON
    let parsed;
    try {
      parsed = JSON.parse(raw);
    } catch (e) {
      return { valid: false, error: 'JSON_PARSE_ERROR', raw };
    }

    // 步骤 2: Schema 校验 — 检查 required 字段 + enum + 类型
    const schema = roleCfg.response_schema;
    if (schema) {
      const schemaErrors = this._validateSchema(parsed, schema);
      if (schemaErrors.length > 0) {
        return { valid: false, error: 'SCHEMA_MISMATCH', parsed, details: schemaErrors };
      }
    }

    // 步骤 3: 正则黑名单 — 拦截违规模式
    const patterns = roleCfg.output_filter?.forbidden_patterns || [];
    const content = parsed.content || '';
    for (const pattern of patterns) {
      if (new RegExp(pattern, 'i').test(content)) {
        return { valid: false, error: 'FORBIDDEN_PATTERN', parsed, matched: pattern };
      }
    }

    return { valid: true, data: parsed };
  }

  static _validateSchema(obj, schema) {
    const errors = [];
    // required 字段检查
    if (schema.required) {
      for (const field of schema.required) {
        if (!(field in obj)) errors.push(`缺少必填字段: ${field}`);
      }
    }
    // enum 值检查
    if (schema.properties) {
      for (const [key, prop] of Object.entries(schema.properties)) {
        if (prop.enum && obj[key] && !prop.enum.includes(obj[key])) {
          errors.push(`字段 ${key} 值 "${obj[key]}" 不在允许范围内: [${prop.enum.join(', ')}]`);
        }
      }
    }
    return errors;
  }
}
```

### 校验失败后的重试策略

```javascript
// background.js — 模型输出不合法时自动重试
async function safeGenerate(modelCfg, roleCfg, userMessage, maxRetries = 2) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const raw = await callOllama(modelCfg, roleCfg, userMessage);
    const result = OutputFilter.validate(roleCfg, raw);

    if (result.valid) return result.data;

    // 重试时在 prompt 中附加错误信息，告诉模型哪里错了
    if (attempt < maxRetries) {
      userMessage = `你的上一次回复格式不正确: ${result.error}。请严格按照要求的 JSON 格式重新回答。原始问题: ${userMessage}`;
    } else {
      throw new Error(`模型输出校验失败，已重试 ${maxRetries} 次: ${result.error}`);
    }
  }
}
```

---

## 五、关键技术实现

### 1. Connection Check — Ollama 可用性检测

> **v2 新增**：审查员指出，Ollama 未运行时不能静默失败。

```javascript
// lib/connection-check.js
class ConnectionCheck {
  static async ping() {
    try {
      const res = await fetch('http://localhost:11434/api/tags', {
        signal: AbortSignal.timeout(3000)
      });
      return res.ok ? { status: 'online' } : { status: 'error' };
    } catch {
      return { status: 'offline' };
    }
  }

  // 获取当前加载的模型列表 + 状态
  static async loadedModels() {
    const res = await fetch('http://localhost:11434/api/ps');
    const data = await res.json();
    return data.models || []; // [{ name, size, expires_at, ... }]
  }
}
```

### 2. 流式输出（v2 增强 NDJSON 边界处理）

> **v2 修订**：审查员指出 NDJSON 解析有边界情况（chunk 切割在行中间、空行、`[DONE]` 标记），这是项目最复杂的技术点。

```javascript
// lib/ollama-client.js
class OllamaClient {
  /**
   * 流式调用 Ollama，返回 AsyncGenerator
   * 处理了 NDJSON 的边界情况：跨 chunk 行分割、空行、JSON 解析失败容错
   */
  static async *streamChat(model, messages, format = 'json') {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s 总体超时

    try {
      const res = await fetch('http://localhost:11434/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, messages, stream: true, format }),
        signal: controller.signal
      });

      if (!res.ok) {
        const errBody = await res.text();
        throw new Error(`Ollama API 错误 ${res.status}: ${errBody}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // 追加到 buffer，按行切分，最后不完整的行留在 buffer 中
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // 最后一行可能不完整

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;          // 跳过空行
          if (trimmed === '[DONE]') break; // 结束标记

          try {
            const parsed = JSON.parse(trimmed);
            if (parsed.message?.content) {
              yield parsed.message.content;
            }
            if (parsed.done) return;
          } catch {
            // 个别行 JSON 损坏时静默跳过，不中断整个流
            console.warn('跳过损坏的 NDJSON 行:', trimmed.substring(0, 80));
          }
        }
      }
    } finally {
      clearTimeout(timeoutId);
    }
  }
}
```

### 3. Service Worker 保活（v2 增强）

```javascript
// background.js
// Manifest V3 的 SW 30s 无活动会被杀 — 三种策略结合：
// ① chrome.alarms 心跳每 25s 触发一次
// ② 流式请求期间持有 port 连接（SW 不会在有活跃连接时被终止）
// ③ 调用 Ollama 前先发 ping 确认存活

chrome.alarms.create('keepalive', { periodInMinutes: 0.4 }); // ~25s
chrome.alarms.onAlarm.addListener(() => {
  console.debug('[SW] keepalive @', new Date().toISOString());
});

// 长时间 inactivity 后的自我保护
let lastActivity = Date.now();
setInterval(() => {
  if (Date.now() - lastActivity > 600_000) { // 10分钟无活动
    console.debug('[SW] 长时间无活动，主动休眠');
    // chrome.alarms 心跳继续保持，需要时可被唤醒
  }
}, 60_000);
```

### 4. 请求队列 + 并发控制

```javascript
// lib/request-queue.js
class RequestQueue {
  constructor(maxConcurrent = 1) {
    this.queue = [];
    this.running = 0;
    this.maxConcurrent = maxConcurrent;
  }

  enqueue(fn, timeoutMs = 30000) {
    return new Promise((resolve, reject) => {
      this.queue.push({ fn, resolve, reject, timeoutMs });
      this._process();
    });
  }

  async _process() {
    if (this.running >= this.maxConcurrent || this.queue.length === 0) return;
    this.running++;
    const { fn, resolve, reject, timeoutMs } = this.queue.shift();
    try {
      const result = await Promise.race([
        fn(),
        new Promise((_, r) => setTimeout(() => r(new Error('请求超时')), timeoutMs))
      ]);
      resolve(result);
    } catch (e) {
      reject(e);
    }
    this.running--;
    this._process();
  }

  get pending() { return this.queue.length; }
}
```

### 5. XSS 防护 — Shadow DOM + textContent

```javascript
// content/agent-panel.js
class AgentPanel {
  constructor() {
    this.host = document.createElement('div');
    this.host.id = 'local-agent-root';
    // mode: 'closed' = 页面 JS 无法通过 element.shadowRoot 访问内部
    this.shadow = this.host.attachShadow({ mode: 'closed' });
    this._buildUI();
    document.body.appendChild(this.host);
  }

  _buildUI() {
    this.shadow.innerHTML = `
      <style>
        :host { all: initial; }  /* 隔离外部 CSS 污染 */
        .panel { position: fixed; bottom: 20px; right: 20px; z-index: 2147483647; }
        /* ... */
      </style>
      <div class="panel">
        <div class="agent-status" id="status">检查连接中...</div>
        <div class="agent-messages" id="messages"></div>
        <input class="agent-input" id="input" placeholder="输入问题..." />
      </div>
    `;
    // 事件绑定等
  }

  appendMessage(text) {
    const div = document.createElement('div');
    div.className = 'message';
    div.textContent = text;  // ← 始终用 textContent，绝不用 innerHTML！
    this.shadow.getElementById('messages').appendChild(div);
  }

  // 逐 token 追加（流式打字机效果）
  appendToken(token) {
    let lastMsg = this.shadow.querySelector('.message.streaming');
    if (!lastMsg) {
      lastMsg = document.createElement('div');
      lastMsg.className = 'message streaming';
      this.shadow.getElementById('messages').appendChild(lastMsg);
    }
    lastMsg.textContent += token;
  }

  setStatus(status) {
    const statusEl = this.shadow.getElementById('status');
    const statusMap = {
      online: '✅ 模型就绪',
      loading: '⏳ 模型加载中...',
      offline: '❌ Ollama 未运行 — 请启动 Ollama',
      error: '⚠️ 连接异常'
    };
    statusEl.textContent = statusMap[status] || status;
  }
}
```

---

## 六、项目结构

```
extension/
├── manifest.json              # MV3, 最小权限原则
├── config/
│   ├── models.json            # 模型注册 (num_ctx, temperature, max_tokens)
│   ├── roles.json             # 可复用角色模板 (system_prompt, output_filter, schema)
│   └── pages.json             # 页面-角色绑定 (URL pattern → model_ref + role_ref)
├── background.js              # SW 主入口
├── lib/
│   ├── prompt-builder.js      # Prompt 模板拼装
│   ├── ollama-client.js       # Ollama HTTP 封装 (chat + streamChat)
│   ├── request-queue.js       # 请求队列 (多标签页并发控制)
│   ├── output-filter.js       # 三层校验 (JSON parse + schema + 正则)
│   └── connection-check.js    # Ollama 存活探测 + 模型状态查询
├── content/
│   ├── content.js             # 页面注入入口
│   ├── content.css            # 面板样式
│   └── agent-panel.js         # Shadow DOM 面板组件 (含状态展示)
└── popup/
    ├── popup.html             # 模型状态 + 快速切换 + 首次使用引导
    ├── popup.js
    └── popup.css
```

### `manifest.json`

```json
{
  "manifest_version": 3,
  "name": "本地模型智能体",
  "version": "0.1.0",
  "description": "将本地 Ollama 模型嵌入任意网页作为智能体",
  "permissions": ["storage", "alarms", "activeTab"],
  "host_permissions": ["http://localhost:11434/*"],
  "background": {
    "service_worker": "background.js"
  },
  "content_scripts": [{
    "matches": ["<all_urls>"],
    "js": ["content/content.js"],
    "css": ["content/content.css"]
  }],
  "action": {
    "default_popup": "popup/popup.html"
  },
  "web_accessible_resources": [{
    "resources": ["config/*.json"],
    "matches": ["<all_urls>"]
  }]
}
```

**权限最小化说明**：
- `storage`：本地对话历史存储
- `alarms`：Service Worker 保活心跳
- `activeTab`：用户点击触发，不需要 `<all_urls>` 大权限
- `host_permissions` 限 `localhost:11434`，不会访问任何外网

---

## 七、实施路线图（v2 修订）

> **v2 修订**：审查员评估原估时 5 天偏乐观，调整为 7-8 天。P2 增加配置加载开销，P4 拆为两个子阶段。

| 阶段 | 内容 | 产出 | 估时 |
|------|------|------|:--:|
| **P1** 最小原型 | Ollama `/api/chat` 直连 + 硬编码角色 + 简单聊天 UI | 可对话的浮动面板（无配置、无约束） | 1天 |
| **P2** 配置驱动 | JSON 配置加载 + URL glob 匹配 + model_ref/role_ref 解析 | 换模型、换角色只需改 JSON | 1.5-2天 |
| **P3** 三层约束 | output-filter（JSON parse + schema 校验 + 正则）+ Shadow DOM + XSS 防护 + 重试机制 | 模型输出被锁死在 schema 内 | 1.5天 |
| **P4a** 流式输出 | Ollama `stream: true` + ReadableStream + NDJSON 边界处理 + port 通信 + 逐 token 渲染 | 打字机效果，长回复不卡顿 | 1天 |
| **P4b** 健壮性 | 请求队列 + SW 保活 + 连接检测 + Ollama 离线降级 + 模型状态 + 首次使用引导 | 多标签页、模型冷启动不崩 | 1天 |
| **P5** 记忆管理 | localStorage 对话历史 + 每会话 50 轮上限 + LRU 淘汰 + 多会话切换 | 上下文连续，不爆存储 | 0.5天 |
| **P6** 测试验证 | 各模型的 JSON Schema 符合率测试 + 边界场景 + Bug 修复 | 稳定可用 | 0.5-1天 |
| **P7** 打磨发布 | popup 配置面板完善 + 打包脚本 + 用户文档 | 可分发 | 0.5天 |
| **总计** | | | **7-8.5天** |

### Phase 依赖关系

```
P1 ──→ P2 ──→ P3 ──→ P4a ──→ P4b ──→ P5 ──→ P6 ──→ P7
```

P3 和 P4a 可以部分并行（Shadow DOM 在 P3，流式通信在 P4a，不冲突）。

---

## 八、技术选型对照（含 v2 变更）

| 选项 | 选择 | 替代方案 | 原因 |
|------|:--:|------|------|
| 模型运行时 | **Ollama** | llama.cpp / vLLM / LM Studio | 标准 HTTP API + 一键安装 + Windows 原生支持 |
| 嵌入方式 | **Browser Extension** | 独立 Electron 应用 / iframe | 天然嵌入任意网页，低侵入 |
| AI 框架 | **无**（手写 HTTP 调用） | LangChain.js | Ollama API 足够简单，LangChain 过度抽象 |
| 配置格式 | **JSON** (v2 变更) | YAML | 浏览器零依赖，`fetch` + `JSON.parse` 原生支持 |
| 角色复用 | **roles.json 独立文件** (v2 新增) | 内嵌在 pages 中 | 避免多页面共用角色时的重复配置 |
| 约束机制 | **System Prompt + output-filter(Json+Schema+Regex) + Ollama format:json** | LangChain guardrails / Function Calling | 三层渐进，第二层承担 Schema 校验重任 |
| 记忆存储 | **localStorage（起步）** | IndexedDB / chrome.storage | 实现简单，后续可升级 |
| 通信方式 | **chrome.runtime + port** | WebSocket / postMessage | 浏览器原生 API，无需额外基础设施 |
| 中文模型 | **qwen2.5:7b** (首选) | deepseek-r1 / glm4 / yi | 中文能力最均衡，JSON 格式遵循率最高 |

---

## 九、用户前置条件

唯一需要做的事：

```bash
# 1. 下载安装 Ollama (Windows)
# https://ollama.com/download/windows

# 2. 拉取中文模型（推荐 qwen2.5:7b，JSON 遵循率最高）
ollama pull qwen2.5:7b

# 3. 验证 Ollama 运行正常
ollama run qwen2.5:7b "你好，请回复JSON格式: {\"status\": \"ok\"}"

# 4. 加载浏览器扩展
# chrome://extensions/ → 开发者模式 → 加载已解压的扩展 → 选择 extension/ 目录
```

之后完全离线运行，无需任何网络。

---

## 十、已知风险 & 缓解措施

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 7b 模型 JSON Schema 符合率仅 80-85% | 校验失败需重试，响应变慢 | ① few-shot example 提升符合率 ② 最多重试 2 次 ③ 小模型专用宽松 schema |
| DeepSeek-R1 `<think>` 标签泄漏到 JSON | JSON 解析失败 | output-filter 的 JSON.parse 捕获后触发重试 |
| Ollama 模型冷启动 10-30 秒 | 用户首次请求超时 | popup 显示模型状态 + 请求超时设为 60s（而非 30s） |
| Service Worker 被杀 | 长推理被打断 | chrome.alarms 心跳 + 流式连接持 port |
| 多标签页同时请求 | 后续请求超时 | 请求队列 + 友好的"模型正忙"提示 |
| 对话历史撑爆 localStorage | 扩展不可用 | 50 轮上限 + LRU 淘汰，P5 阶段落实 |

---

> **审查记录**  
> v1 → 审查员 #1: 架构确认，指出 System Prompt 偏软 / XSS 风险 / SW 生命周期 / 流式必要性 / 估时偏乐观  
> v2: 整合所有反馈 — JSON 替代 YAML、roles.json 角色复用、output-filter 加强到 Schema 校验、P4 拆分为两个子阶段、总估时调整为 7-8 天、新增连接检测和模型状态展示、manifest 权限最小化
