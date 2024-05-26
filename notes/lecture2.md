# 基础作业🛝
## 使用 InternLM2-Chat-1.8B 模型生成 300 字的小故事🍑🍑🍑
![](../images/lecture2/1716703350655.png)
## 找到书生·浦语 1 处表现不佳的案例并提交问卷🍒🍒🍒
![](../images/lecture2/1716704732036.png)
![](../images/lecture2/1716704910167.png)

# 进阶作业🎢
## 熟悉huggingface下载功能🍓🍓🍓
![](../images/lecture2/1716708410563.png)
![](../images/lecture2/1716716022345.png)
## 完成 浦语·灵笔2 的 图文创作 及 视觉问答 部署🫐🫐🫐
图文创作：
![](../images/lecture2/1716715759638.png)
![](../images/lecture2/1716715794188.png)
![](../images/lecture2/1716715833944.png)

![](../images/lecture2/1716716220702.png)
![](../images/lecture2/1716716233752.png)
![](../images/lecture2/1716716247108.png)

视觉问答：
![](../images/lecture2/1716716856318.png)
![](../images/lecture2/1716716888210.png)
## 完成 Lagent 工具调用 数据分析 Demo 部署🥝🥝🥝
![](../images/lecture2/1716718906951.png)

# Lagent初识
Lagent 是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体， 并提供了一些典型工具为大语言模型赋能。它的整个框架图如下:
![](../images/lecture2/1716723325036.png)

Lagent 的特性总结如下：

- 流式输出：提供 stream_chat 接口作流式输出，本地就能演示酷炫的流式 Demo。
- 接口统一，设计全面升级，提升拓展性，包括：
  - Model : 不论是 OpenAI API, Transformers 还是推理加速框架 LMDeploy 一网打尽，模型切换可以游刃有余；
  - Action: 简单的继承和装饰，即可打造自己个人的工具集，不论 InternLM 还是 GPT 均可适配；
  - Agent：与 Model 的输入接口保持一致，模型到智能体的蜕变只需一步，便捷各种 agent 的探索实现；
- 文档全面升级，API 文档全覆盖。

