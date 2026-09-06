import sys

from rhubarb.cli_client import run_prompt


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Reply with exactly the word: pong"
    response = run_prompt(prompt)
    print(response["result"])


if __name__ == "__main__":
    main()
