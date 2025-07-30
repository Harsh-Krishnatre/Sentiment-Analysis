from model.train import train_model
from model.test import test_model

def menu():
    while True:
        print("\n Sentiment Analysis Model")
        print("1. Train Model")
        print("2. Test Model")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            train_model()
        elif choice == "2":
            test_model()
        elif choice == "3":
            print("Bye.")
            break
        else:
            print(" Invalid option. Try again.")

if __name__ == "__main__":
    menu()
