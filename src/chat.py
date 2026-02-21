from search import RAGSearch

def main():
    rag = RAGSearch()
    print("\n" + "=" * 50)
    print("Bem-vindo ao Chat com Busca Vetorial (RAG)")
    print("Digite 'sair' para encerrar\n")
    print("=" * 50)
    
    while True:
        pergunta = input("\nFaça sua pergunta: ").strip()
        
        if pergunta.lower() in ["sair", "exit", "quit"]:
            print("\nAté logo!")
            break
        
        if not pergunta:
            print("Por favor, digite uma pergunta válida.")
            continue
        
        resposta = rag.generate_answer(pergunta)
        print("\n" + "=" * 50)
        print(f"Pergunta: {pergunta}")
        print("=" * 50)
        print(f"Resposta: {resposta}")
        print("=" * 50)

if __name__ == "__main__":
    main()