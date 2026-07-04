from agents.student_agent import student_chat
from agents.professor_agent import professor_chat

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