from search import RAGSearch

def main():
    while True:
        pergunta = input("\nFaça sua pergunta: ").strip()
        resposta = RAGSearch().generate_answer(pergunta)
        print("\n" + "=" * 40)
        print(f"Pergunta: {pergunta}")
        print("=" * 40)
        print(f"Resposta: {resposta}")
        print("=" * 40)

if __name__ == "__main__":
    main()