from src.rag_engine.preprocessing.hashing import content_hash


text_a = "Apple released a new product."
text_b = "Apple released a new product."
text_c = "Apple launched a new product."


print("A:", content_hash(text_a))
print("B:", content_hash(text_b))
print("C:", content_hash(text_c))

print("\nA == B:", content_hash(text_a) == content_hash(text_b))
print("A == C:", content_hash(text_a) == content_hash(text_c))