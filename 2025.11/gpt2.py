#!/usr/bin/env python3
"""
GPT-2 Web GUI 版本 (使用 Gradio)
使用方法：
    pip install gradio
    python gpt2_web.py
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr


class GPT2WebApp:
    def __init__(self, model_name='gpt2'):
        print(f"🔄 加载模型: {model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✅ 模型已加载到 {self.device}")

    def generate(self,
                 prompt,
                 max_length,
                 temperature,
                 top_k,
                 top_p,
                 num_sequences,
                 repetition_penalty,
                 use_sampling):
        """生成文本"""
        if not prompt.strip():
            return "⚠️ 请输入 Prompt！"

        try:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature if use_sampling else 1.0,
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_sequences,
                    repetition_penalty=repetition_penalty,
                    do_sample=use_sampling,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            results = []
            for i, output in enumerate(outputs, 1):
                text = self.tokenizer.decode(output, skip_special_tokens=True)
                results.append(f"{'=' * 70}\n【生成 {i}】\n{'=' * 70}\n{text}\n")

            return "\n".join(results)

        except Exception as e:
            return f"❌ 错误: {str(e)}"


# 创建应用
app = GPT2WebApp('gpt2')

# 创建 Gradio 界面
with gr.Blocks(title="GPT-2 文本生成器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 GPT-2 文本生成器

    输入提示文本（Prompt），调整参数，点击"生成"按钮开始创作！
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # 输入区域
            prompt_input = gr.Textbox(
                label="📝 输入 Prompt",
                placeholder="Once upon a time...",
                lines=5
            )

            generate_btn = gr.Button("🚀 生成文本", variant="primary", size="lg")

            gr.Markdown("---")
            gr.Markdown("### ⚙️ 生成参数")

            # 基础参数
            with gr.Accordion("基础参数", open=True):
                max_length = gr.Slider(
                    minimum=20,
                    maximum=500,
                    value=100,
                    step=10,
                    label="📏 最大长度 (max_length)",
                    info="生成文本的最大token数量"
                )

                num_sequences = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="🔢 生成数量 (num_sequences)",
                    info="生成几个不同的版本"
                )

            # 采样参数
            with gr.Accordion("采样参数", open=True):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    label="🌡️ 温度 (temperature)",
                    info="控制随机性：越低越保守，越高越创意"
                )

                top_k = gr.Slider(
                    minimum=10,
                    maximum=100,
                    value=50,
                    step=5,
                    label="🎯 Top-K",
                    info="从概率最高的K个词中采样"
                )

                top_p = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.95,
                    step=0.05,
                    label="🎲 Top-P (Nucleus)",
                    info="从累积概率达到P的词中采样"
                )

            # 高级参数
            with gr.Accordion("高级参数", open=False):
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="🔁 重复惩罚 (repetition_penalty)",
                    info="避免重复：1.0 = 不惩罚，越大惩罚越重"
                )

                use_sampling = gr.Checkbox(
                    value=True,
                    label="🎰 启用采样 (do_sample)",
                    info="禁用则使用贪心解码（确定性输出）"
                )

            # 预设配置
            gr.Markdown("---")
            gr.Markdown("### 🎨 快速预设")

            with gr.Row():
                preset_creative = gr.Button("✍️ 创意写作", size="sm")
                preset_factual = gr.Button("📰 事实陈述", size="sm")
                preset_code = gr.Button("💻 代码生成", size="sm")

        with gr.Column(scale=1):
            # 输出区域
            output_text = gr.Textbox(
                label="✨ 生成结果",
                lines=25,
                max_lines=30
            )

    # 示例
    gr.Markdown("---")
    gr.Markdown("### 💡 示例 Prompt")

    gr.Examples(
        examples=[
            ["Once upon a time in a magical forest", 100, 1.0, 50, 0.95, 2],
            ["The future of artificial intelligence is", 120, 0.8, 50, 0.9, 3],
            ["In a world where technology has advanced beyond imagination", 150, 1.2, 50, 0.95, 2],
            ["The secret to happiness is", 80, 0.7, 40, 0.9, 3],
            ["def calculate_fibonacci(n):", 100, 0.3, 30, 0.85, 1],
        ],
        inputs=[prompt_input, max_length, temperature, top_k, top_p, num_sequences],
        label="点击示例快速开始"
    )

    # 绑定生成按钮
    generate_btn.click(
        fn=app.generate,
        inputs=[
            prompt_input,
            max_length,
            temperature,
            top_k,
            top_p,
            num_sequences,
            repetition_penalty,
            use_sampling
        ],
        outputs=output_text
    )


    # 预设按钮功能
    def apply_creative_preset():
        return 1.2, 50, 0.95, 1.2, True


    def apply_factual_preset():
        return 0.7, 40, 0.9, 1.0, True


    def apply_code_preset():
        return 0.3, 30, 0.85, 1.0, True


    preset_creative.click(
        fn=apply_creative_preset,
        outputs=[temperature, top_k, top_p, repetition_penalty, use_sampling]
    )

    preset_factual.click(
        fn=apply_factual_preset,
        outputs=[temperature, top_k, top_p, repetition_penalty, use_sampling]
    )

    preset_code.click(
        fn=apply_code_preset,
        outputs=[temperature, top_k, top_p, repetition_penalty, use_sampling]
    )

    # 使用说明
    with gr.Accordion("📖 使用说明", open=False):
        gr.Markdown("""
        ## 参数说明

        ### 🌡️ Temperature (温度)
        - **0.1-0.5**: 保守、可预测（适合事实性内容、代码）
        - **0.6-0.9**: 平衡、自然（适合一般写作）
        - **1.0-1.5**: 创意、随机（适合创意写作、头脑风暴）
        - **1.5+**: 极度随机（实验性）

        ### 🎯 Top-K
        - 从概率最高的 K 个词中随机选择
        - 越小越保守，越大越多样

        ### 🎲 Top-P (Nucleus Sampling)
        - 从累积概率达到 P 的词中选择
        - 0.9-0.95 是常用值

        ### 🔁 Repetition Penalty
        - 1.0 = 不惩罚重复
        - 1.2-1.5 = 轻度惩罚（推荐）
        - 2.0 = 强力惩罚

        ### 💡 使用技巧
        - 创意写作：高温度 + 高 Top-P
        - 事实陈述：低温度 + 低 Top-K
        - 代码生成：极低温度 + 禁用采样
        """)

# 启动应用
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 启动 Web 界面...")
    print("=" * 70)
    demo.launch(
        share=False,  # 设为 True 可生成公开链接
        server_name="0.0.0.0",
        server_port=7860
    )