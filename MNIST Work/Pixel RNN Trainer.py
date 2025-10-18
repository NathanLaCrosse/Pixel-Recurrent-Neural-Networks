import MNISTClassifier as classifier
import MNISTGen as gen

'''
  The purpose of this file is to train multiple different networks back to back
without human intervention, to be run on the main workstation...
'''

def main():
    while True:
        try:
            print("\nTrain Generation (1)")
            print("Train Pixel RNN (2)")
            print("Quit (q)")
            raw = input("Enter 1, 2, or q: ").strip().lower()

            if raw in {"q", "quit", "exit"}:
                print("End")
                break

            try:
                choice = int(raw)
            except ValueError:
                print("Invalid input. Please enter 1, 2, or q.")
                continue

            if choice == 1:
                print(">>> Running Generation Training...")
                gen.train_generation(epochs=1, batch_size=100, model_file_name='Gen TEST.pt')
                gen.generator('Gen TEST.pt')

            elif choice == 2:
                print(">>> Running Pixel RNN Training...")
                classifier.train_pixel_rnn(epochs=1, model_name="Test.pt")
                classifier.evaluate(train_network=None, dataset=None, filepath='Test.pt')
                for _ in range(10):
                    classifier.manual_evaluation(filepath='Test.pt')

            else:
                print("Invalid choice. Please enter 1, 2, or q.")

        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()