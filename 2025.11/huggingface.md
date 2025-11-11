Huggingface使用教程，pytorch版本都可以使用，主要用于下载对应的模型进行调用
pipeline()：利用预训练模型进行推理的最简单方式

例子：pipeline(task=“sentiment-analysis”)，情感分析任务

# 交互式 GPT-2 文本生成器

以下是一个带交互式界面的 GPT-2 文本生成类，支持自定义生成参数（温度、Top-K 等）、多轮生成和参数动态修改：

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class InteractiveGPT2:
    def __init__(self, model_name='gpt2'):
        print("🚀 加载 GPT-2 模型...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"✅ 模型已加载到 {self.device}\n")
    
    def generate(self, prompt, **kwargs):
        """生成文本"""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def run(self):
        """运行交互式界面"""
        print("=" * 70)
        print("🤖 GPT-2 文本生成器")
        print("=" * 70)
        print("\n指令:")
        print("  - 输入文本开始生成")
        print("  - 输入 'settings' 修改参数")
        print("  - 输入 'quit' 退出")
        print("\n默认参数:")
        
        # 默认参数
        params = {
            'max_length': 100,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.95,
            'num_return_sequences': 1,
            'do_sample': True
        }
        
        for key, value in params.items():
            print(f"  {key}: {value}")
        print()
        
        while True:
            try:
                user_input = input("\n💬 输入 Prompt (或指令): ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    print("👋 再见！")
                    break
                
                if user_input.lower() == 'settings':
                    print("\n⚙️  当前参数:")
                    for key, value in params.items():
                        print(f"  {key}: {value}")
                    
                    print("\n修改参数 (直接回车保持不变):")
                    for key in params.keys():
                        new_value = input(f"  {key} [{params[key]}]: ").strip()
                        if new_value:
                            try:
                                if key == 'do_sample':
                                    params[key] = new_value.lower() == 'true'
                                elif key == 'num_return_sequences':
                                    params[key] = int(new_value)
                                elif key == 'max_length' or key == 'top_k':
                                    params[key] = int(new_value)
                                else:
                                    params[key] = float(new_value)
                            except:
                                print(f"  ⚠️  无效值，保持 {key}={params[key]}")
                    continue
                
                # 生成文本
                print("\n🔄 生成中...")
                
                for i in range(params['num_return_sequences']):
                    result = self.generate(user_input, **params)
                    print(f"\n📝 生成 {i+1}:")
                    print("-" * 70)
                    print(result)
                    print("-" * 70)
            
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    generator = InteractiveGPT2('gpt2')
    generator.run()
```

学习了如何使用huggingface的模型简单调用，实现如情感分类，文本生成等效果，还需回顾相关代码，网址中有相应的进阶教程，如微调等技术，需要仔细阅读代码学习

gpt2.py实现了一个网站形式的自动化脚本运行，可以输入相应内容进行输出，可以看作一个简单的demo学习。
