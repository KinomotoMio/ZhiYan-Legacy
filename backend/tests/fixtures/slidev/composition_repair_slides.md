---
theme: default
title: Claude Code Prompt Suggestions 背后的设计哲学
layout: cover
class: deck-cover
transition: slide-left
---

# Prompt，不只是文字

说出来，就是做出来。

> Prompt 是指令，不是描述
> AI 替你写指令 = 替你行动

---
layout: context
class: deck-context
transition: slide-left
---

# 藏在提示词背后的三次转向

**Prompt Suggestions 功能** 的核心指令经历了三次迭代，揭示了截然不同的产品哲学。

| 版本 | 核心指令 | 设计意图 |
|:----:|:---------|:---------|
| v2.0.55 | Predict what the user will type next | 预测用户意图 |
| ??? | ??? | ??? |
| 协作者模式 | 沉默是金 | 隐忍克制 |

> J.L. Austin 的言语行为理论：说出来本身就是一种行动

---
layout: section
class: deck-section
transition: slide-left
---

# PART 01

## v2.0.55: Predict what the user will type next

---
layout: focus-explainer
class: deck-framework
transition: slide-left
---

# 协作者模式：沉默是金

协作者模式的核心指令只有一条触发原则——**用户在看，不在打字**。

```text
当用户正在阅读输出时，不要主动生成下一条提示。
沉默通常是正确答案。
```

- 知道何时闭嘴，比知道说什么更难
- 隐式需求：软件行业的默认常识
- 指令长度：283 tokens

> 施事性话语的本质：说出来就是做出来

---
layout: end
class: deck-closing
transition: slide-left
---

# 四岁小孩就会的事

**Theory of Mind（心理理论）**

> 推测他人信念和意图的能力，通常在四岁左右开始发展

Claude Code 正在学习这件事。

**不是预测用户会打什么字，而是理解用户在想什么。**
