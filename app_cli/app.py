from model_util import model, tokenizer, generate_text

if __name__ == "__main__":
    device = "cpu"
    money = input("구입금액을 입력해 주세요.\n").strip()
    winning = input("당첨 번호를 입력해 주세요.\n").strip()
    bonus = input("보너스 번호를 입력해 주세요.\n").strip()
    example_prompt = (
        f"money={money}\n"
        f"winning={winning}\n"
        f"bonus={bonus}\n"
        "###\n"
    )
    out = generate_text(model, tokenizer, example_prompt, device=device)
    print(out)