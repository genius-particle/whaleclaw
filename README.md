# WhaleClaw

WhaleClaw 是一个基于 Python 与 AI 编程工具协同开发的本地 Agent 项目。

首先感谢 [OpenClaw](https://github.com/openclaw/openclaw)。WhaleClaw 的很多核心思路来自 OpenClaw：本地优先、工具驱动、多渠道接入、可扩展记忆与会话管理。在此基础上，WhaleClaw 做了更偏工程化和成本控制的实现，目标是“可长期运行、可控 token、可解释行为”。

## 项目定位

WhaleClaw 是一个运行在本地的 AI Agent 框架，支持 WebChat 与飞书等渠道，具备工具调用、会话管理、长期记忆、定时任务与插件扩展能力。它不是“只会聊天”的模型包装层，而是“消息 -> 决策 -> 工具 -> 结果 -> 记忆沉淀”的完整闭环系统。

## 核心功能

- 多模型路由：支持 Anthropic、OpenAI、DeepSeek、Qwen、GLM、Gemini 等。
- 多渠道接入：WebChat、飞书（持续扩展）。
- 工具体系：bash、文件读写编辑、浏览器、会话工具、技能工具等。
- 会话管理：持久化消息、上下文窗口裁剪、会话级 token 统计。
- 长期记忆：自动捕获、检索召回、LLM 组织、全局风格沉淀。
- 可观测性：结构化日志与可追踪的调用链。

## 架构概览

WhaleClaw 采用“Gateway + Agent Loop + Store + Tool Registry”架构：

1. Gateway 接收 Web/API/WS 请求并维护会话状态。  
2. Agent Loop 负责推理循环：组装提示词、调用模型、执行工具、写回结果。  
3. Session Store 负责会话消息与统计持久化。  
4. Memory Manager 负责长期记忆的 recall/capture/organize。  
5. Tool Registry 统一管理可用工具与 schema。  

这套结构借鉴 OpenClaw 的运行时思想，但更强调“低耦合 + 低 token 常态开销”。

## 重点：WhaleClaw 如何省 Token

相对 OpenClaw 常见的“工具说明 + 记忆 + 技能”重注入，WhaleClaw 采取分层与预算策略：

- 分层提示词：静态层、动态层、延迟层，避免全量拼接。  
- 工具 schema 走 API 参数：尽量不把工具说明灌进 prompt。  
- 技能按需路由：只注入命中的技能，不做全量技能展开。  
- 记忆有预算：recall 有 token 上限，先短画像再细节。  
- 召回有门控：普通闲聊不查记忆，创作任务优先注入 L0。  
- 风格常驻短注入：把稳定风格沉淀成极短 system 指令，替代高频 recall。  

简单说：WhaleClaw 不是“每轮都把所有上下文塞给模型”，而是“最小必要信息注入”。

## 重点：长期记忆机制

WhaleClaw 的长期记忆不是单点工具，而是完整流水线：

1. **自动捕获**：识别“偏好/默认/以后/请记住”等信号，写入候选记忆。  
2. **去重与限频**：避免重复写入和高频噪声。  
3. **缓冲批量写入**：弱信号先缓冲，强信号可立即落库。  
4. **LLM 组织器**：周期性把 raw 记忆整理为 L0/L1 画像。  
5. **全局风格提炼**：从稳定偏好提取 `style_directive`，每轮低成本应用。  

同时提供 `/api/memory/style`，可在 WebChat 设置里查看、手动覆盖、清除全局风格，做到“自动优先，人工可控”。

## 重点：会话压缩机制

WhaleClaw 的会话压缩目标是“保留语义、缩减成本、不中断对话”：

- 通过上下文窗口保护最近关键消息，避免误裁剪。  
- 老消息在后台压缩为摘要（分层摘要），减少历史 token 占用。  
- 会话压缩与长期记忆协同：历史事实可沉淀到记忆，避免重复携带。  

相比 OpenClaw 常见的大上下文硬顶，WhaleClaw 更强调“压缩前移、记忆承接、实时预算”。

## 与 OpenClaw 的关系

WhaleClaw 不是对 OpenClaw 的否定，而是针对“长期运行成本与本地可控性”的工程化延伸：

- 保留 OpenClaw 的 agent 工具链思想。  
- 强化 token 预算与分层注入。  
- 强化“自动记忆 -> 组织 -> 全局风格”闭环。  
- 提供更直接的可操作接口与前端设置入口。  

再次感谢 OpenClaw 社区提供的优秀基础与启发。

注：飞书机器人权限，直接复制粘贴即可
{
  "scopes": {
    "tenant": [
      "cardkit:card:write",
      "contact:contact.base:readonly",
      "contact:user.employee_id:readonly",
      "im:message",
      "im:message.group_at_msg:readonly",
      "im:message.group_msg",
      "im:message.p2p_msg:readonly",
      "im:message.reactions:write_only",
      "im:message:readonly",
      "im:message:send_as_bot",
      "im:message:send_sys_msg",
      "im:resource"
    ],
    "user": []
  }
}


mac 一键包 链接：https://pan.quark.cn/s/901b031c8b47
win 一键包 链接：https://pan.quark.cn/s/425bc7996fa4
