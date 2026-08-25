from research_agents.professor_agent import professor_chat
from research_agents.student_agent import student_chat
from tools.load_data import ensure_chroma_loaded


def main():
    ensure_chroma_loaded()
    while True:
        print("\n===== Research Matching Chatbot =====")
        print("1. Student")
        print("2. Professor")
        print("3. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            student_chat()

        elif choice == "2":
            professor_chat()

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid Choice")


if __name__ == "__main__":
    main()
