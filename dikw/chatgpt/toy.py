import os
import openai
# 从环境变量中获取 OpenAI API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def generate_text(prompt):
    """
    使用 OpenAI API 生成文本。
    """
    try:
        # 调用 OpenAI API，生成文本
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=1024,
            n=1,
            stop=None
        )

        # 返回生成的文本
        return response.choices[0].text

    except Exception as e:
        # 处理异常情况
        print(f"生成文本失败：{e}")
        return None

if __name__ == "__main__":
    # 示例用法
    prompt = "如何三天入门python"
    generated_text = generate_text(prompt)
    print(generated_text)