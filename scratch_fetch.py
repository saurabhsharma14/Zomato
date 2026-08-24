from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset("ManikaSaini/zomato-restaurant-recommendation", split="train")

print("Features:")
print(dataset.features)
print("\nFirst row:")
print(dataset[0])
